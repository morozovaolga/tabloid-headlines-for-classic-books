"""
Генератор жёлтых заголовков для русской классики
Пайплайн: HuggingFace Dataset → Preprocessing → Ollama → Headlines
"""

import requests
import json
import re
import time
import random
from typing import List, Dict, Optional, Tuple
from datasets import load_dataset

# ========== КОНФИГУРАЦИЯ ==========

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"  # Или "gemma2:2b" / "mistral:7b"

OUTPUT_FILE = "yellow_headlines.jsonl"
BOOKS_PER_RUN = 10  # Количество рандомных книг за раз

# ========== ШАГ 1: ЗАГРУЗКА ДАННЫХ ИЗ HUGGINGFACE ==========

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


# ========== ШАГ 2: ГЕНЕРАЦИЯ ЗАГОЛОВКОВ ==========

def build_prompt(title: str, author: str, summary: str) -> str:
    """Улучшенный промпт с указанием типа"""
    
    author_str = f"автор: {author}" if author else "классическое произведение"
    
    return f"""Ты — редактор жёлтой газеты. Создай ОДИН скандальный заголовок для ЛИТЕРАТУРНОГО ПРОИЗВЕДЕНИЯ.

ВАЖНО: Это художественная литература (роман/повесть/пьеса), НЕ биография реального человека!

ПРАВИЛА:
- Заголовок про СЮЖЕТ книги, не про автора или реальных людей
- Используй обобщённые описания персонажей: "молодой человек", "девушка", "офицер"
- НЕ используй имена персонажей! Только социальные роли
- Заголовок должен быть драматичным и скандальным
- Формат: 6-10 слов
- Без кавычек и метаданных

ЗАПРЕЩЕНО:
❌ Имена персонажей (Раскольников, Наташа, Болконский)
❌ Упоминание автора в заголовке
❌ Мета-формулировки ("в романе...", "герой книги...")

Произведение: {title} ({author_str})

Сюжет:
{summary}

Создай ТОЛЬКО заголовок, без пояснений:"""

def call_ollama(prompt: str) -> Optional[str]:
    """Запрос к Ollama с обработкой ошибок"""
    
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.9,
                    "top_p": 0.95,
                }
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "").strip()
        
    except Exception as e:
        print(f"  ⚠️ Ошибка Ollama: {e}")
    
    return None

def clean_headline(text: str) -> str:
    """Очистка заголовка от мусора"""
    
    # Убрать префиксы
    prefixes = [
        r'^Заголовок:\s*',
        r'^Ответ:\s*',
        r'^\d+\.\s*',
        r'^[-*•]\s*'
    ]
    
    for prefix in prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE)
    
    # Убрать кавычки
    text = text.strip('"«»"\'')
    
    # Взять только первую строку
    text = text.split('\n')[0]
    
    # Убрать лишние пробелы
    text = ' '.join(text.split())
    
    return text

def score_headline(headline: str, summary: str) -> float:
    """Оценка качества заголовка (0-1)"""
    
    score = 1.0
    headline_lower = headline.lower()
    
    # Штрафы
    penalties = {
        # Упоминание метаданных
        r'\bроман\b|\bповесть\b|\bпьеса\b|\bкнига\b': 0.3,
        r'\bгерой\b|\bперсонаж\b|\bгероиня\b': 0.2,
        
        # Слишком литературно
        r'\bпроизведение\b|\bсочинение\b|\bтворение\b': 0.4,
        
        # Имена собственные (скорее всего персонажи)
        r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b': 0.5,
        
        # Слишком длинный
        'len_over_12': 0.2 if len(headline.split()) > 12 else 0,
        
        # Слишком короткий
        'len_under_5': 0.3 if len(headline.split()) < 5 else 0,
    }
    
    for pattern, penalty in penalties.items():
        if pattern.startswith('len_'):
            score -= penalty
        elif re.search(pattern, headline_lower):
            score -= penalty
    
    # Бонусы
    bonuses = {
        # Драматические слова
        r'скандал|шок|трагед|драм|секрет|тайн|изме|убий|смерть': 0.15,
        
        # Социальные роли (хорошо!)
        r'офицер|студент|дворянин|князь|купец|врач|учитель': 0.1,
        
        # Обобщённые персонажи
        r'молод(?:ой|ая)|девушк|мужчин|женщин|парень': 0.1,
    }
    
    for pattern, bonus in bonuses.items():
        if re.search(pattern, headline_lower):
            score += bonus
    
    return max(0.0, min(1.0, score))

def generate_best_headline(title: str, author: str, summary: str, num_attempts: int = 5) -> Tuple[Optional[str], float]:
    """Генерация с выбором лучшего из N попыток"""
    
    candidates = []
    
    for attempt in range(num_attempts):
        # Генерация
        prompt = build_prompt(title, author, summary)
        raw_headline = call_ollama(prompt)
        
        if not raw_headline:
            continue
        
        # Очистка
        headline = clean_headline(raw_headline)
        
        # Оценка
        score = score_headline(headline, summary)
        
        candidates.append((headline, score))
        
        print(f"     Попытка {attempt+1}: {headline} (score: {score:.2f})")
    
    if not candidates:
        return None, 0.0
    
    # Выбрать лучший
    best_headline, best_score = max(candidates, key=lambda x: x[1])
    
    print(f"  ✅ Выбран лучший: {best_headline} (score: {best_score:.2f})")
    
    return best_headline, best_score

def generate_fallback_headline(title: str, author: str, summary: str) -> Optional[str]:
    """Резервный генератор на основе шаблонов"""
    
    summary = summary.lower()
    
    # Определить профессию/роль
    profession_patterns = {
        'врач|доктор': 'Врач',
        'писатель|автор|литератор': 'Писатель',
        'князь|княгиня': 'Аристократ',
        'дворянин|барин|помещик': 'Помещик',
        'студент': 'Студент',
        'офицер|военный|солдат': 'Военный',
        'чиновник|служащий': 'Чиновник',
        'купец|торговец': 'Купец',
    }
    
    profession = None
    for pattern, prof in profession_patterns.items():
        if re.search(pattern, summary, re.IGNORECASE):
            profession = prof
            break
    
    if not profession:
        return None
    
    # Шаблоны с сохранением исторического контекста
    drama_templates = {
        'убил|убийство|дуэль|застрелил': [
            f"{profession} совершил убийство: подробности трагедии",
            f"Кровавая драма: {profession.lower()} довёл ситуацию до смерти",
        ],
        'пропил|спился|алкоголь': [
            f"{profession} спился и разорил семью",
            f"Алкоголь разрушил жизнь {profession.lower()}а: эксклюзив",
        ],
        'развод|измена|бросил.*жену|оставил.*мужа': [
            f"{profession} разрушил семью ради страсти",
            f"Любовный треугольник довёл {profession.lower()}а до разрыва",
        ],
        'война|сражение|бой|ранен': [
            f"{profession} пережил военную трагедию: откровения",
            f"Война изменила жизнь {profession.lower()}а навсегда",
        ],
        'имение|усадьба|сад|продать|вырубить': [
            f"Родовое поместье продано с молотка: подробности",
            f"{profession} не смог спасти семейное гнездо от разорения",
        ],
    }
    
    for pattern, templates in drama_templates.items():
        if re.search(pattern, summary, re.IGNORECASE):
            return random.choice(templates)
    
    defaults = [
        f"{profession} оказался в центре скандала: что случилось",
        f"Трагическая судьба {profession.lower()}а: эксклюзивные подробности",
    ]
    
    return random.choice(defaults)

# ========== ОСНОВНОЙ ПАЙПЛАЙН ==========

def process_books(books: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Полный пайплайн с отслеживанием пропущенных"""
    
    results = []
    skipped = []
    
    print("="*70)
    print("ГЕНЕРАТОР ЖЁЛТЫХ ЗАГОЛОВКОВ С АВТООЦЕНКОЙ КАЧЕСТВА")
    print("="*70)
    print(f"\nВсего книг к обработке: {len(books)}\n")
    
    for i, book in enumerate(books, 1):
        title = book["title"]
        author = book["author"]
        
        print(f"\n[{i}/{len(books)}] {title}")
        print("-" * 70)
        print(f"  Автор: {author or 'не указан'}")
        
        # Подготовить текст
        summary = prepare_book_text(book)
        
        if len(summary) < 100:
            skipped.append({
                "title": title,
                "reason": f"Текст слишком короткий ({len(summary)} символов)"
            })
            print(f"  ⚠️ Пропущено: недостаточно текста\n")
            continue
        
        print(f"  Текст: {len(summary)} символов")
        
        # Генерация
        print(f"\n  🤖 Генерация заголовка...")
        headline, score = generate_best_headline(
            title, 
            author, 
            summary,
            num_attempts=5
        )
        
        if not headline:
            skipped.append({
                "title": title,
                "reason": "Не удалось сгенерировать заголовок"
            })
            print(f"  ❌ Не удалось сгенерировать заголовок\n")
            continue
        
        # Сохранить результат
        result = {
                "input": f"Название: {title}\n{summary}",
            "target": "",
            "meta": {
                "id": book["id"],
                "title": title,
                "author": author,
                "score": round(score, 2)
            },
            "suggestions": [headline]
        }
        
        results.append(result)
        
        time.sleep(1)
    
    return results, skipped

def save_results(results: List[Dict], filename: str, skipped_books: List[Dict] = None):
    """Сохранить результаты + список пропущенных"""
    
    # JSONL формат
    with open(filename, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    # JSON для удобства просмотра
    json_filename = filename.replace('.jsonl', '.json')
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Сохранить список пропущенных
    if skipped_books:
        skipped_filename = filename.replace('.jsonl', '_skipped.txt')
        with open(skipped_filename, "w", encoding="utf-8") as f:
            f.write("ПРОПУЩЕННЫЕ КНИГИ (требуют ручной обработки):\n\n")
            for i, book in enumerate(skipped_books, 1):
                f.write(f"{i}. {book['title']}\n")
                f.write(f"   Причина: {book['reason']}\n\n")
        
        print(f"   - {skipped_filename} (пропущенные книги)")
    
    # Статистика
    if results:
        avg_score = sum(r['meta'].get('score', 0) for r in results) / len(results)
        high_quality = sum(1 for r in results if r['meta'].get('score', 0) >= 0.7)
        
        print(f"\n{'='*70}")
        print(f"📊 СТАТИСТИКА КАЧЕСТВА:")
        print(f"   Средняя оценка: {avg_score:.2f}")
        print(f"   Высокое качество (≥0.7): {high_quality}/{len(results)} ({high_quality/len(results)*100:.1f}%)")
        print(f"{'='*70}")
    
    print(f"\n{'='*70}")
    print(f"✅ Результаты сохранены:")
    print(f"   - {filename} (JSONL формат)")
    print(f"   - {json_filename} (JSON для просмотра)")
    print(f"✅ Обработано книг: {len(results)}")
    
    if skipped_books:
        print(f"⚠️  Пропущено книг: {len(skipped_books)}")
    
    print(f"{'='*70}\n")

# ========== ЗАПУСК ==========

if __name__ == "__main__":
    # Проверка что Ollama запущена
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        print(f"✅ Ollama работает, модель: {OLLAMA_MODEL}\n")
    except:
        print("❌ ОШИБКА: Ollama не запущена!")
        print("Запусти сервер: ollama serve")
        print(f"Скачай модель: ollama pull {OLLAMA_MODEL}\n")
        exit(1)
    
    # Загрузка случайных книг из датасета
    books = load_random_books(num_books=BOOKS_PER_RUN)
    
    if not books:
        print("❌ Не удалось загрузить книги из датасета")
        exit(1)
    
    # Обработка книг
    results, skipped = process_books(books)
    
    # Сохранение
    save_results(results, OUTPUT_FILE, skipped_books=skipped)
    
    # Показать лучшие примеры
    if results:
        print("\n📋 ЛУЧШИЕ ЗАГОЛОВКИ:\n")
        
        # Отсортировать по score
        sorted_results = sorted(results, key=lambda x: x['meta'].get('score', 0), reverse=True)
        
        for result in sorted_results[:10]:
            title = result['meta']['title']
            headline = result['suggestions'][0]
            score = result['meta'].get('score', 0)
            
            print(f"📖 {title} (score: {score:.2f})")
            print(f"   → {headline}\n")
