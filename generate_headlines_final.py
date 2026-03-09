"""
Генератор жёлтых заголовков для русской классики
Пайплайн: HuggingFace Dataset → RAG → Ollama → Headlines

УЛУЧШЕНИЯ:
✅ Использует датасет slon-hk/BooksSummarizationRU
✅ RAG для поиска похожих примеров
✅ Автоматическая оценка качества
✅ Выбор лучшего из N попыток
"""

import requests
import json
import re
import time
import random
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ========== КОНФИГУРАЦИЯ ==========

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

OUTPUT_FILE = "yellow_headlines_rag.jsonl"
EXAMPLES_DB = "examples_db.json"
BOOKS_PER_RUN = 10

# ========== RAG СИСТЕМА ==========

class HeadlineRAG:
    """Retrieval-Augmented Generation для заголовков"""
    
    def __init__(self, examples_file: str = EXAMPLES_DB):
        """Загрузить базу примеров и создать индекс"""
        
        try:
            with open(examples_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.examples = data['examples']
            
            print(f"✅ Загружено {len(self.examples)} примеров из {examples_file}")
            
        except FileNotFoundError:
            print(f"⚠️ Файл {examples_file} не найден. Создаю базовую базу примеров...")
            self.examples = self._create_default_examples()
            self._save_examples(examples_file)
        
        # Создать TF-IDF индекс
        self._build_index()
    
    def _create_default_examples(self) -> List[Dict]:
        """Создать базовую базу качественных примеров"""
        
        return [
            {
                "book": "Преступление и наказание",
                "author": "Достоевский",
                "plot_keywords": ["студент", "убийство", "старуха", "топор", "совесть", "бедность", "преступление"],
                "headline": "Студент скрывал страшную тайну: что довело его до отчаяния?",
                "score": 0.95
            },
            {
                "book": "Анна Каренина",
                "author": "Толстой",
                "plot_keywords": ["замужняя", "офицер", "любовь", "поезд", "самоубийство", "измена", "страсть"],
                "headline": "Роковая страсть довела светскую даму до трагедии",
                "score": 0.92
            },
            {
                "book": "Евгений Онегин",
                "author": "Пушкин",
                "plot_keywords": ["дворянин", "дуэль", "друг", "убийство", "скука", "поэт", "ревность"],
                "headline": "Дворянин не простил оскорбления: дуэль закончилась трагедией",
                "score": 0.88
            },
            {
                "book": "Отцы и дети",
                "author": "Тургенев",
                "plot_keywords": ["нигилист", "дворянка", "дуэль", "смерть", "конфликт", "молодёжь", "заражение"],
                "headline": "Молодой нигилист заплатил страшную цену за убеждения",
                "score": 0.85
            },
            {
                "book": "Война и мир",
                "author": "Толстой",
                "plot_keywords": ["война", "офицер", "ранение", "плен", "любовь", "наполеон", "сражение"],
                "headline": "Война изменила жизнь офицера навсегда: откровения из плена",
                "score": 0.82
            },
            {
                "book": "Мастер и Маргарита",
                "author": "Булгаков",
                "plot_keywords": ["дьявол", "писатель", "балл", "полёт", "безумие", "москва", "сделка"],
                "headline": "Женщина пошла на сделку с дьяволом: что произошло дальше?",
                "score": 0.90
            },
            {
                "book": "Вишнёвый сад",
                "author": "Чехов",
                "plot_keywords": ["имение", "продажа", "разорение", "сад", "вырубка", "дворянство", "долги"],
                "headline": "Родовое имение продано с молотка: вишнёвый сад вырубят",
                "score": 0.87
            },
            {
                "book": "Герой нашего времени",
                "author": "Лермонтов",
                "plot_keywords": ["офицер", "дуэль", "похищение", "кавказ", "горянка", "смерть", "скандал"],
                "headline": "Офицер похитил княжну: скандал потряс кавказское общество",
                "score": 0.86
            },
            {
                "book": "Идиот",
                "author": "Достоевский",
                "plot_keywords": ["князь", "купец", "красавица", "убийство", "ревность", "безумие", "треугольник"],
                "headline": "Любовный треугольник довёл купца до страшного поступка",
                "score": 0.89
            },
            {
                "book": "Мёртвые души",
                "author": "Гоголь",
                "plot_keywords": ["чиновник", "махинация", "обман", "души", "скандал", "авантюра", "помещики"],
                "headline": "Чиновник провернул дерзкую аферу: как его разоблачили?",
                "score": 0.84
            }
        ]
    
    def _save_examples(self, filename: str):
        """Сохранить базу примеров"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump({"examples": self.examples}, f, ensure_ascii=False, indent=2)
        
        print(f"✅ База примеров сохранена в {filename}")
    
    def _build_index(self):
        """Построить TF-IDF индекс для поиска"""
        
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=1000,
            min_df=1
        )
        
        # Индексировать ключевые слова примеров
        example_texts = [' '.join(ex['plot_keywords']) for ex in self.examples]
        self.example_vectors = self.vectorizer.fit_transform(example_texts)
        
        print(f"✅ TF-IDF индекс построен: {len(self.examples)} примеров")
    
    def find_similar(self, plot_text: str, top_k: int = 3) -> List[Dict]:
        """Найти наиболее похожие примеры"""
        
        # Векторизовать сюжет
        plot_vector = self.vectorizer.transform([plot_text.lower()])
        
        # Вычислить косинусное сходство
        similarities = cosine_similarity(plot_vector, self.example_vectors)[0]
        
        # Получить top_k индексов
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Собрать результаты
        results = []
        for idx in top_indices:
            example = self.examples[idx].copy()
            example['similarity'] = round(float(similarities[idx]), 3)
            results.append(example)
        
        return results
    
    def add_example(self, book: str, author: str, keywords: List[str], 
                   headline: str, score: float):
        """Добавить новый пример в базу"""
        
        new_example = {
            "book": book,
            "author": author,
            "plot_keywords": keywords,
            "headline": headline,
            "score": score
        }
        
        self.examples.append(new_example)
        self._build_index()
        self._save_examples(EXAMPLES_DB)
        
        print(f"✅ Добавлен пример: {headline} (score: {score:.2f})")

# ========== ЗАГРУЗКА ДАТАСЕТА ==========

def load_random_books(num_books: int = 10) -> List[Dict]:
    """Загрузить случайные книги из датасета HuggingFace"""
    
    print(f"📚 Загрузка датасета slon-hk/BooksSummarizationRU...")
    
    try:
        dataset = load_dataset("slon-hk/BooksSummarizationRU", split="train")
        
        print(f"✅ Датасет загружен: {len(dataset)} книг")
        
        # Выбрать случайные индексы
        total_books = len(dataset)
        random_indices = random.sample(range(total_books), min(num_books, total_books))
        
        books = []
        for idx in random_indices:
            item = dataset[idx]
            
            # Извлечь данные
            book_data = {
                "title": item.get("Title", ""),
                "author": item.get("Author", ""),
                "summary": item.get("Summary", ""),
                "id": f"hf_{idx}"
            }
            
            # Проверка что есть краткое содержание
            summary_text = book_data["summary"] or ""
            if len(summary_text.strip()) > 500:
                books.append(book_data)
        
        print(f"✅ Отобрано {len(books)} книг с достаточным объёмом текста\n")
        return books
        
    except Exception as e:
        print(f"❌ Ошибка загрузки датасета: {e}")
        return []

def prepare_book_text(book: Dict) -> str:
    """Подготовить текст книги для промпта"""
    
    text = book.get("summary", "")
    
    # Ограничение для оптимального качества
    if len(text) > 2000:
        text = text[:2000] + "..."
    
    return text

# ========== ГЕНЕРАЦИЯ С RAG ==========

def build_rag_prompt(title: str, author: str, plot: str, rag: HeadlineRAG) -> str:
    """Промпт с динамическими примерами из RAG"""
    
    # Найти похожие примеры
    similar = rag.find_similar(plot, top_k=3)
    
    # Сформировать секцию примеров
    examples_text = "✨ ПРИМЕРЫ ОТЛИЧНЫХ ЗАГОЛОВКОВ (похожие ситуации):\n\n"
    
    for i, ex in enumerate(similar, 1):
        sim_pct = ex['similarity'] * 100
        examples_text += f"{i}. \"{ex['headline']}\"\n"
        examples_text += f"   ({ex['book']}, похожесть: {sim_pct:.0f}%, качество: {ex['score']:.2f})\n\n"
    
    # Основной промпт
    author_str = f"автор: {author}" if author else "классика"
    
    prompt = f"""Ты — редактор жёлтой газеты. Создай ОДИН интригующий заголовок для литературного произведения.

{examples_text}

📋 ЗОЛОТОЕ ПРАВИЛО ЖЁЛТОЙ ПРЕССЫ:
ИНТРИГУЙ, НЕ РАСКРЫВАЙ! Заголовок должен вызывать вопрос "Что же произошло?", а не отвечать на него.

✅ ПРАВИЛЬНО (создаёт интригу):
- "Роковая страсть довела офицера до отчаяния: шокирующие подробности"
- "Студент совершил страшное: что скрывала его совесть?"
- "Светская львица приняла роковое решение: трагедия в высшем свете"
- "Он думал, что тайна останется нераскрытой, но..."
- "Дворянин не выдержал позора: чем закончился скандал?"

❌ НЕПРАВИЛЬНО (раскрывает сюжет):
- "Студент убил старуху топором" (весь сюжет раскрыт!)
- "Замужняя дама бросилась под поезд" (концовка испорчена!)
- "Дворянин застрелил друга на дуэли" (нет интриги!)

✅ ПРИЁМЫ ИНТРИГИ:
- Недосказанность: "...довела до трагедии", "...принял роковое решение"
- Вопросы: "Что скрывал [роль]?", "Чем обернулась [ситуация]?"
- Последствия вместо действий: "...шокировал общество" вместо "убил"
- Эмоции: "не выдержал", "довела до отчаяния", "роковая страсть"
- Двоеточия и многоточия для паузы: "Трагедия в семье: подробности..."

✅ ИСПОЛЬЗУЙ:
- Социальные роли: студент, офицер, дворянин, купец, врач
- Слова интриги: тайна, скандал, трагедия, роковой, страшный, шокирующий
- Формулы: "[Роль] сделал [что-то страшное]: что произошло дальше?"
- 8-12 слов для создания напряжения

❌ ЗАПРЕЩЕНО:
- Раскрывать развязку (убил, застрелился, бросилась)
- Имена персонажей (Раскольников, Онегин, Анна)
- Слова "роман", "герой", "героиня", "произведение", "книга"

Произведение: {title} ({author_str})

Краткое содержание:
{plot}

Создай ОДИН интригующий заголовок (заинтригуй, НЕ раскрывай концовку):"""
    
    return prompt

def call_ollama(prompt: str) -> Optional[str]:
    """Запрос к Ollama"""
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.85,
                    "top_p": 0.9,
                }
            },
            timeout=45
        )
        
        if response.status_code == 200:
            return response.json().get("response", "").strip()
    
    except Exception as e:
        print(f"  ⚠️ Ошибка Ollama: {e}")
    
    return None

def clean_headline(text: str) -> str:
    """Очистка заголовка"""
    
    # Убрать префиксы
    text = re.sub(r'^(Заголовок|Ответ|Вариант):\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+\.\s*', '', text)
    text = re.sub(r'^[-*•]\s*', '', text)
    
    # Убрать кавычки
    text = text.strip('"«»"\'')
    
    # Только первая строка
    text = text.split('\n')[0]
    
    # Убрать лишние пробелы
    text = ' '.join(text.split())
    
    return text

def score_headline(headline: str) -> float:
    """Оценка качества заголовка (0-1)"""
    
    score = 1.0
    hl = headline.lower()
    
    # Штрафы
    penalties = {
        r'\bроман\b|\bповесть\b|\bгерой\b|\bгероиня\b|\bкнига\b': 0.3,
        r'\bпроизведение\b|\bсочинение\b': 0.4,
        r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+)?\b': 0.5,  # Имена
    }
    
    for pattern, penalty in penalties.items():
        if re.search(pattern, hl):
            score -= penalty
    
    # Бонусы
    if re.search(r'скандал|шок|трагед|драм|секрет|убий|смерть|изме|афер|разор', hl):
        score += 0.15
    
    if re.search(r'офицер|студент|дворянин|купец|врач|князь|чиновник', hl):
        score += 0.1
    
    if re.search(r'молод(?:ой|ая)|девушк|мужчин|женщин', hl):
        score += 0.05
    
    # Длина
    word_count = len(headline.split())
    if word_count < 5:
        score -= 0.3
    elif word_count > 12:
        score -= 0.2
    elif 6 <= word_count <= 10:
        score += 0.1
    
    return max(0.0, min(1.0, score))

def generate_with_rag(title: str, author: str, plot: str, rag: HeadlineRAG, 
                     num_attempts: int = 5) -> Tuple[Optional[str], float, List[Dict]]:
    """Генерация с RAG и выбором лучшего"""
    
    candidates = []
    
    for attempt in range(num_attempts):
        # Промпт с примерами
        prompt = build_rag_prompt(title, author, plot, rag)
        
        # Генерация
        raw = call_ollama(prompt)
        if not raw:
            continue
        
        # Очистка и оценка
        headline = clean_headline(raw)
        score = score_headline(headline)
        
        candidates.append((headline, score))
        print(f"     #{attempt+1}: {headline} (оценка: {score:.2f})")
    
    if not candidates:
        return None, 0.0, []
    
    # Лучший вариант
    best_headline, best_score = max(candidates, key=lambda x: x[1])
    
    # Найти использованные примеры для логирования
    used_examples = rag.find_similar(plot, top_k=3)
    
    print(f"  ✅ Выбран лучший: {best_headline} (оценка: {best_score:.2f})")
    
    return best_headline, best_score, used_examples

# ========== ОСНОВНОЙ ПАЙПЛАЙН ==========

def process_books_with_rag(books: List[Dict], rag: HeadlineRAG) -> Tuple[List[Dict], List[Dict]]:
    """Обработка книг с RAG"""
    
    results = []
    skipped = []
    
    print("\n" + "="*70)
    print("ГЕНЕРАТОР ЖЁЛТЫХ ЗАГОЛОВКОВ С RAG")
    print("="*70)
    print(f"\nВсего книг: {len(books)}")
    print(f"Примеров в базе RAG: {len(rag.examples)}\n")
    
    for i, book in enumerate(books, 1):
        title = book["title"]
        author = book["author"]
        
        print(f"\n[{i}/{len(books)}] {title}")
        print("-" * 70)
        print(f"  Автор: {author or 'не указан'}")
        
        # Подготовить текст
        plot = prepare_book_text(book)
        
        if len(plot) < 500:
            skipped.append({
                "title": title,
                "reason": f"Текст слишком короткий ({len(plot)} символов)"
            })
            print(f"  ⚠️ Пропущено: недостаточно текста\n")
            continue
        
        print(f"  Текст: {len(plot)} символов")
        
        # Генерация с RAG
        print(f"\n  🤖 Генерация с RAG (5 попыток)...")
        headline, score, used_examples = generate_with_rag(
            title, 
            author, 
            plot,
            rag,
            num_attempts=5
        )
        
        if not headline:
            skipped.append({
                "title": title,
                "reason": "Не удалось сгенерировать заголовок"
            })
            print(f"  ❌ Генерация не удалась\n")
            continue
        
        # Сохранить результат
        result = {
            "input": f"Название: {title}\nАвтор: {author}\n\n{plot}",
            "target": "",
            "meta": {
                "id": book["id"],
                "title": title,
                "author": author,
                "score": round(score, 2),
                "rag_examples_used": [ex['book'] for ex in used_examples]
            },
            "suggestions": [headline]
        }
        
        results.append(result)
        
        # Автоматически добавить отличные примеры в базу
        if score >= 0.85:
            print(f"  🌟 Отличное качество! Добавляю в базу RAG...")
            # Извлечь ключевые слова (простой метод)
            keywords = re.findall(r'\b[а-яё]{4,}\b', plot.lower())
            keywords = list(set(keywords))[:10]  # Топ-10 уникальных
            
            rag.add_example(title, author, keywords, headline, score)
        
        time.sleep(1)
    
    return results, skipped

def save_results(results: List[Dict], filename: str, skipped_books: List[Dict] = None):
    """Сохранить результаты"""
    
    # JSONL формат
    with open(filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # JSON для просмотра
    json_filename = filename.replace('.jsonl', '.json')
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Пропущенные
    if skipped_books:
        skipped_filename = filename.replace('.jsonl', '_skipped.txt')
        with open(skipped_filename, "w", encoding="utf-8") as f:
            f.write("ПРОПУЩЕННЫЕ КНИГИ:\n\n")
            for i, book in enumerate(skipped_books, 1):
                f.write(f"{i}. {book['title']}\n")
                f.write(f"   Причина: {book['reason']}\n\n")
    
    # Статистика
    if results:
        avg_score = sum(r['meta']['score'] for r in results) / len(results)
        high_quality = sum(1 for r in results if r['meta']['score'] >= 0.7)
        excellent = sum(1 for r in results if r['meta']['score'] >= 0.85)
        
        print(f"\n{'='*70}")
        print(f"📊 СТАТИСТИКА:")
        print(f"   Средняя оценка: {avg_score:.2f}")
        print(f"   Высокое качество (≥0.7): {high_quality}/{len(results)} ({high_quality/len(results)*100:.1f}%)")
        print(f"   Отличное качество (≥0.85): {excellent}/{len(results)} ({excellent/len(results)*100:.1f}%)")
        print(f"{'='*70}")
    
    print(f"\n✅ Результаты сохранены:")
    print(f"   - {filename}")
    print(f"   - {json_filename}")
    if skipped_books:
        print(f"   - {skipped_filename}")
    print(f"\n✅ Обработано: {len(results)} книг")
    if skipped_books:
        print(f"⚠️  Пропущено: {len(skipped_books)} книг")
    print(f"{'='*70}\n")

# ========== ЗАПУСК ==========

if __name__ == "__main__":
    # Проверка Ollama
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"✅ Ollama работает, модель: {OLLAMA_MODEL}\n")
    except:
        print("❌ ОШИБКА: Ollama не запущена!")
        print("Запусти: ollama serve")
        print(f"Скачай модель: ollama pull {OLLAMA_MODEL}\n")
        exit(1)
    
    # Инициализация RAG
    print("="*70)
    print("ИНИЦИАЛИЗАЦИЯ RAG СИСТЕМЫ")
    print("="*70 + "\n")
    
    rag = HeadlineRAG()
    
    # Загрузка книг
    books = load_random_books(num_books=BOOKS_PER_RUN)
    
    if not books:
        print("❌ Не удалось загрузить книги")
        exit(1)
    
    # Обработка с RAG
    results, skipped = process_books_with_rag(books, rag)
    
    # Сохранение
    save_results(results, OUTPUT_FILE, skipped_books=skipped)
    
    # Показать лучшие
    if results:
        print("\n📋 ТОП-10 ЛУЧШИХ ЗАГОЛОВКОВ:\n")
        
        sorted_results = sorted(results, key=lambda x: x['meta']['score'], reverse=True)
        
        for i, result in enumerate(sorted_results[:10], 1):
            title = result['meta']['title']
            headline = result['suggestions'][0]
            score = result['meta']['score']
            
            print(f"{i}. {title} (оценка: {score:.2f})")
            print(f"   → {headline}\n")
