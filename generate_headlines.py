"""
Генератор жёлтых заголовков для русской классики
Пайплайн: Wikipedia → Preprocessing → Ollama → Headlines
"""

import wikipedia
import requests
import json
import re
import time
from typing import List, Dict, Optional, Tuple

# ========== КОНФИГУРАЦИЯ ==========

wikipedia.set_lang('ru')

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"  # Или "gemma2:2b" / "mistral:7b"

OUTPUT_FILE = "yellow_headlines.jsonl"

# ========== СПИСОК КНИГ ==========

RUSSIAN_BOOKS = [
    "А. Т. Твардовский. Василий Теркин",
    "А. П. Платонов. Возвращение",
    "У. Шекспир. Ромео и Джульетта",
    "Мольер. Мещанин во дворянстве",
    "В. Скотт. Айвенго",
    "Д.Д. Сэлинджер. Над пропастью во ржи",
    "Ф. М. Достоевский. Преступление и наказание",
    "Л. Н. Толстой. Война и мир",
    "Л. Н. Толстой. Анна Каренина",
    "Л. Н. Толстой. Богатырь из Белева",
    "Л. Н. Толстой. Детство",
    "Л. Н. Толстой. Кавказский пленник",
    "Л. Н. Толстой. Казаки",
]

# ========== ШАГ 1: СБОР ДАННЫХ ИЗ ВИКИПЕДИИ ==========

def fetch_book_from_wikipedia(title: str) -> Optional[Dict]:
    """Получить данные книги из Википедии с умным поиском"""
    
    # Если в названии нет уточнения, пробуем добавить автоматически
    search_variants = [title]
    
    if not any(x in title.lower() for x in ['роман', 'повесть', 'пьеса', 'рассказ']):
        search_variants.extend([
            f"{title} (роман)",
            f"{title} (повесть)",
            f"{title} (пьеса)"
        ])
    
    for search_title in search_variants:
        try:
            # Сначала пробуем поиск
            search_results = wikipedia.search(search_title, results=5)
            
            # Фильтруем результаты — ищем литературные произведения
            for result in search_results:
                result_lower = result.lower()
                
                # Пропускаем нелитературные страницы
                skip_keywords = ['певица', 'актриса', 'футболист', 'политик', 
                                'фильм', 'сериал', 'альбом', 'песня']
                if any(kw in result_lower for kw in skip_keywords):
                    continue
                
                # Проверяем что это литература
                good_keywords = ['роман', 'повесть', 'пьеса', 'рассказ', 
                               'произведение', 'книга', 'сборник']
                if any(kw in result_lower for kw in good_keywords):
                    try:
                        page = wikipedia.page(result)
                        
                        # Дополнительная проверка содержимого
                        content_start = page.content[:500].lower()
                        
                        # Это литература?
                        literature_markers = ['роман', 'повесть', 'пьеса', 'написан', 
                                             'произведение', 'книга', 'автор', 'сюжет']
                        if any(marker in content_start for marker in literature_markers):
                            
                            # Извлечь автора
                            author = extract_author(page.content)
                            
                            # Очистить контент
                            clean_content = clean_wikipedia_content(page.content)
                            
                            # Извлечь сюжет
                            plot = extract_plot_section(clean_content)
                            
                            return {
                                "title": page.title,
                                "author": author,
                                "content": clean_content,
                                "plot": plot,
                                "url": page.url,
                                "page_id": page.pageid
                            }
                    except:
                        continue
            
        except wikipedia.exceptions.DisambiguationError as e:
            # Если несколько вариантов, ищем литературный
            for option in e.options:
                option_lower = option.lower()
                
                if any(x in option_lower for x in ['роман', 'повесть', 'пьеса', 'рассказ']):
                    try:
                        return fetch_book_from_wikipedia(option)
                    except:
                        continue
        
        except wikipedia.exceptions.PageError:
            continue
        
        except Exception as e:
            continue
    
    return None

def clean_wikipedia_content(content: str) -> str:
    """Очистка контента от служебных секций"""
    
    sections_to_remove = [
        r'== Примечания ==.*',
        r'== Литература ==.*',
        r'== Ссылки ==.*',
        r'== См\. также ==.*',
        r'== Издания ==.*'
    ]
    
    for pattern in sections_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    return content.strip()

def extract_plot_section(content: str) -> str:
    """Извлечь секцию Сюжет"""
    
    # Попытка найти секцию "Сюжет"
    plot_match = re.search(
        r'== Сюжет ==(.*?)(?===|$)', 
        content, 
        re.DOTALL
    )
    
    if plot_match:
        plot = plot_match.group(1).strip()
    else:
        # Альтернативные названия секций
        for section_name in ["Содержание", "Краткое содержание", "Описание"]:
            plot_match = re.search(
                f'== {section_name} ==(.*?)(?===|$)', 
                content, 
                re.DOTALL
            )
            if plot_match:
                plot = plot_match.group(1).strip()
                break
        else:
            # Если нет секции сюжет, берём начало статьи
            paragraphs = content.split('\n\n')
            plot = '\n\n'.join(paragraphs[:3])
    
    # Очистка
    plot = re.sub(r'\[\d+\]', '', plot)  # Убрать сноски
    
    # Ограничить длину
    if len(plot) > 800:
        plot = plot[:800] + '...'
    
    return plot

def extract_author(content: str) -> str:
    """Улучшенное извлечение автора"""
    
    # Ищем в первых 800 символах
    text_start = content[:800]
    
    patterns = [
        r'(?:роман|повесть|пьеса|рассказ|произведение)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})',
        r'написан(?:а|о)?\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})',
        r'автор\s*[—-]\s*([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})',
        r'([А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+)\s+написал',
        r'—\s+(?:роман|повесть|пьеса)\s+([А-ЯЁ][а-яё]+(?:\s+[А-ЯЁ][а-яё]+){1,2})',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text_start)
        if match:
            author = match.group(1).strip()
            
            # Очистка окончаний творительного падежа
            author = re.sub(r'ым$', 'ий', author)
            author = re.sub(r'им$', 'ий', author)
            author = re.sub(r'ой$', 'ая', author)
            
            # Проверка что это не название места
            bad_words = ['Москва', 'Россия', 'СССР', 'Советск']
            if not any(bad in author for bad in bad_words):
                return author
    
    return ""

# ========== ШАГ 2: ГЕНЕРАЦИЯ ЗАГОЛОВКОВ ==========

def build_prompt(title: str, author: str, plot: str) -> str:
    """Улучшенный промпт с указанием типа"""
    
    author_str = f"автор: {author}" if author else "классическое произведение"
    
    return f"""Ты — редактор жёлтой газеты. Создай ОДИН скандальный заголовок для ЛИТЕРАТУРНОГО ПРОИЗВЕДЕНИЯ.

ВАЖНО: Это художественная литература (роман/повесть/пьеса), НЕ биография реального человека!

ПРАВИЛА:
- Заголовок про СЮЖЕТ книги, не про автора или реальных людей
- Основан на событиях из художественного произведения
- Стиль современной новостной сенсации
- НЕ искажай исторический контекст (дуэль остаётся дуэлью)
- Обобщай персонажей: "молодой человек", "аристократка", "студент"
- 8-15 слов
- ТОЛЬКО заголовок, без пояснений и кавычек

ХОРОШИЕ ПРИМЕРЫ:

Книга: "Евгений Онегин", автор: Пушкин
Сюжет: Молодой дворянин убивает друга на дуэли из-за флирта с его невестой.
Заголовок: Тусовщик застрелил друга из-за мимолётного флирта

Книга: "Вишнёвый сад", автор: Чехов
Сюжет: Купец предлагает вырубить вишнёвый сад и застроить участок дачами.
Заголовок: Бизнесмен предложил вырубить родовую рощу под застройку

Книга: "Война и мир", автор: Толстой
Сюжет: Раненый князь лежит под дубом и переосмысливает свою жизнь.
Заголовок: Умирающий офицер нашёл смысл жизни под деревом

Книга: "Преступление и наказание", автор: Достоевский
Сюжет: Студент убивает старуху-процентщицу топором, затем мучается от совести.
Заголовок: Студент расправился с кредитором: шокирующие подробности

Книга: "Анна Каренина", автор: Толстой
Сюжет: Замужняя аристократка бросает семью ради романа с офицером.
Заголовок: Светская львица оставила мужа ради молодого любовника

Книга: "Лолита", автор: Набоков
Сюжет: Мужчина средних лет одержим двенадцатилетней падчерицей и увозит её в путешествие.
Заголовок: Отчим похитил несовершеннолетнюю падчерицу: подробности дела

ПЛОХИЕ ПРИМЕРЫ (НЕ ДЕЛАЙ ТАК):

❌ "Набоков написал скандальный роман" — заголовок должен быть про СЮЖЕТ
❌ "Блогер обманул подписчиков" — неуместные современные слова
❌ "Лолита Милявская шокировала публику" — это про певицу, а не про книгу!

ТЕПЕРЬ ТВОЯ ОЧЕРЕДЬ:

Книга: "{title}", {author_str}
ТИП: художественное литературное произведение
Сюжет: {plot}

Заголовок про СЮЖЕТ книги:"""

def generate_headline_ollama(title: str, author: str, plot: str, retry_count: int = 3) -> Optional[str]:
    """Генерация заголовка через Ollama"""
    
    # Проверить что сюжет не пустой
    if not plot or len(plot) < 50:
        print(f"    ⚠️ Сюжет слишком короткий, генерация невозможна")
        return None
    
    prompt = build_prompt(title, author, plot)
    
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.85,
            "num_predict": 60,
            "stop": ["\n", "Книга:", "Сюжет:", "ПРИМЕРЫ"]
        }
    }
    
    for attempt in range(retry_count):
        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
            
            result = response.json()
            headline = result.get("response", "").strip()
            
            headline = clean_generated_headline(headline)
            
            if validate_headline(headline, title, author):
                return headline
            else:
                print(f"    ⚠️ Попытка {attempt + 1}: заголовок не прошёл валидацию")
                continue
        
        except requests.exceptions.ConnectionError:
            print(f"    ❌ Ollama не запущена. Запусти: ollama serve")
            return None
        
        except Exception as e:
            print(f"    ⚠️ Попытка {attempt + 1} ошибка: {e}")
            time.sleep(2)
    
    # Fallback
    print(f"    ⚠️ LLM не сработал, пробуем шаблоны...")
    fallback = generate_template_fallback(title, plot)
    
    if not fallback:
        print(f"    ❌ Шаблоны тоже не сработали (недостаточно данных)")
        return None
    
    return fallback

def clean_generated_headline(text: str) -> str:
    """Очистка сгенерированного текста"""
    
    # Взять только первую строку
    text = text.split('\n')[0].strip()
    
    # Убрать кавычки
    text = text.strip('"«»"\'')
    
    # Убрать "Заголовок:" если модель его вернула
    text = re.sub(r'^(?:Заголовок|Ответ):\s*', '', text, flags=re.IGNORECASE)
    
    # Ограничить длину
    words = text.split()
    if len(words) > 20:
        text = ' '.join(words[:15])
    
    return text.strip()

def validate_headline(headline: str, title: str = "", author: str = "") -> bool:
    """Проверка качества с учётом контекста"""
    
    if not headline or len(headline) < 10:
        return False
    
    # Нет технического мусора
    if any(x in headline for x in ['править', 'код', '[', ']', '<', '>']):
        return False
    
    # Проверка длины
    word_count = len(headline.split())
    if word_count < 5 or word_count > 25:
        return False
    
    # Не должен содержать служебные слова
    if any(x in headline for x in ['Книга:', 'Сюжет:', 'ПРИМЕРЫ', 'автор:']):
        return False
    
    # Не должен содержать фамилию автора в заголовке
    if author:
        author_surnames = author.split()
        for surname in author_surnames:
            if len(surname) > 3 and surname in headline:
                print(f"      ⚠️ Заголовок содержит фамилию автора: {surname}")
                return False
    
    # Не должен быть про автора
    author_phrases = ['написал', 'опубликовал', 'создал роман', 'автор']
    if any(phrase in headline.lower() for phrase in author_phrases):
        print(f"      ⚠️ Заголовок про автора, а не про сюжет")
        return False
    
    return True

# ========== ШАГ 3: ОЦЕНКА КАЧЕСТВА ==========

def score_headline(headline: str, plot: str) -> float:
    """Оценка качества без требования модных словечек"""
    
    score = 0.5
    
    # Длина
    word_count = len(headline.split())
    if 8 <= word_count <= 15:
        score += 0.15
    elif word_count < 6 or word_count > 20:
        score -= 0.2
    
    # Драматичные слова (уместные для любой эпохи)
    drama_words = ['скандал', 'шок', 'трагедия', 'эксклюзив', 'подробности', 
                   'раскрыл', 'признался', 'довёл', 'разрушил', 'шокировал']
    if any(word in headline.lower() for word in drama_words):
        score += 0.15
    
    # МИНУС за анахронизмы (слова которых не было в XIX веке)
    bad_words = ['блогер', 'хейтер', 'инфлюенсер', 'стартап', 'лайк', 
                 'троллинг', 'мем', 'тренд', 'вирусн', 'онлайн']
    if any(word in headline.lower() for word in bad_words):
        score -= 0.3
    
    # МИНУС за неуместную современность
    questionable = ['девелопер', 'менеджер', 'коуч', 'бренд']
    if any(word in headline.lower() for word in questionable):
        score -= 0.15
    
    # ПЛЮС за исторически уместные обобщения
    good_words = ['аристократ', 'помещик', 'офицер', 'чиновник', 'купец',
                  'студент', 'писатель', 'светская львица']
    if any(word in headline.lower() for word in good_words):
        score += 0.1
    
    # Имена собственные (не должно быть)
    if re.search(r'\b[А-ЯЁ][а-яё]+\s+[А-ЯЁ][а-яё]+\b', headline):
        score -= 0.2
    
    # Связь с сюжетом
    plot_keywords = set(re.findall(r'\b\w{4,}\b', plot.lower())[:30])
    headline_words = set(re.findall(r'\b\w{4,}\b', headline.lower()))
    
    if len(plot_keywords & headline_words) >= 2:
        score += 0.15
    
    return max(0.0, min(1.0, score))

def generate_best_headline(title: str, author: str, plot: str, num_attempts: int = 5) -> Tuple[str, float]:
    """Генерация с выбором лучшего варианта по score"""
    
    print(f"  🤖 Генерация {num_attempts} вариантов с оценкой качества...")
    
    candidates = []
    
    for i in range(num_attempts):
        headline = generate_headline_ollama(title, author, plot, retry_count=1)
        
        if headline:
            score = score_headline(headline, plot)
            candidates.append((headline, score))
            
            if score >= 0.5:
                print(f"     ✓ {headline} (score: {score:.2f})")
            else:
                print(f"     ✗ {headline} (score: {score:.2f}) - плохой")
    
    if not candidates:
        print("  ⚠️ Не удалось сгенерировать ни одного варианта, используем fallback")
        fallback = generate_template_fallback(title, plot)
        if fallback:
            return fallback, 0.4
        else:
            return None, 0.0
    
    best_headline, best_score = max(candidates, key=lambda x: x[1])
    
    print(f"\n  ✅ ВЫБРАН ЛУЧШИЙ (score: {best_score:.2f}):\n     → {best_headline}\n")
    
    return best_headline, best_score

# ========== ШАГ 4: FALLBACK ШАБЛОНЫ ==========

def generate_template_fallback(title: str, plot: str) -> Optional[str]:
    """Умные шаблоны если LLM не работает"""
    
    # Проверка что сюжет не пустой
    if not plot or len(plot) < 50:
        return None
    
    import random
    
    # Извлечь профессию/статус
    profession_patterns = {
        'учитель|педагог|географ': 'Учитель',
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
        if re.search(pattern, plot, re.IGNORECASE):
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
        if re.search(pattern, plot, re.IGNORECASE):
            return random.choice(templates)
    
    defaults = [
        f"{profession} оказался в центре скандала: что случилось",
        f"Трагическая судьба {profession.lower()}а: эксклюзивные подробности",
    ]
    
    return random.choice(defaults)

# ========== ОСНОВНОЙ ПАЙПЛАЙН ==========

def process_all_books(books_list: List[str]) -> Tuple[List[Dict], List[Dict]]:
    """Полный пайплайн с отслеживанием пропущенных"""
    
    results = []
    skipped = []
    
    print("="*70)
    print("ГЕНЕРАТОР ЖЁЛТЫХ ЗАГОЛОВКОВ С АВТООЦЕНКОЙ КАЧЕСТВА")
    print("="*70)
    print(f"\nВсего книг к обработке: {len(books_list)}\n")
    
    for i, book_title in enumerate(books_list, 1):
        print(f"\n[{i}/{len(books_list)}] {book_title}")
        print("-" * 70)
        
        # Шаг 1: Получить из Википедии
        print("  📖 Загрузка из Википедии...")
        book_data = fetch_book_from_wikipedia(book_title)
        
        if not book_data:
            skipped.append({
                "title": book_title,
                "reason": "Не найдено в Википедии"
            })
            print(f"  ❌ Пропущено: не найдено в Википедии\n")
            continue
        
        print(f"  ✅ Загружено: {book_data['title']}")
        print(f"     Автор: {book_data['author'] or 'не определён'}")
        print(f"     Сюжет: {len(book_data['plot'])} символов")
        
        # Проверка сюжета
        if len(book_data['plot']) < 100:
            skipped.append({
                "title": book_data['title'],
                "reason": f"Сюжет слишком короткий ({len(book_data['plot'])} символов)"
            })
            print(f"  ⚠️ Пропущено: сюжет слишком короткий\n")
            continue
        
        meaningful_words = len(re.findall(r'\b[а-яА-ЯёЁ]{4,}\b', book_data['plot']))
        if meaningful_words < 20:
            skipped.append({
                "title": book_data['title'],
                "reason": f"Недостаточно информации ({meaningful_words} слов)"
            })
            print(f"  ⚠️ Пропущено: недостаточно информации\n")
            continue
        
        # Шаг 2: Генерация
        print(f"\n  🤖 Генерация заголовка...")
        headline, score = generate_best_headline(
            book_data['title'], 
            book_data['author'], 
            book_data['plot'],
            num_attempts=5
        )
        
        if not headline:
            skipped.append({
                "title": book_data['title'],
                "reason": "Не удалось сгенерировать заголовок"
            })
            print(f"  ❌ Не удалось сгенерировать заголовок\n")
            continue
        
        # Сохранить результат
        result = {
            "input": f"Название: {book_data['title']}\n{book_data['plot']}",
            "target": "",
            "meta": {
                "id": str(book_data['page_id']),
                "title": book_data['title'],
                "author": book_data['author'],
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
    
    # Обработка книг (сначала тест на первых 20)
    results, skipped = process_all_books(RUSSIAN_BOOKS[:20])
    
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