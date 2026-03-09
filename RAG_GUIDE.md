# RAG для генерации заголовков: Полное руководство

## Что такое RAG?

**RAG (Retrieval-Augmented Generation)** = Поиск + Генерация

Вместо того чтобы модель генерировала "из головы", мы:
1. **Ищем похожие примеры** в нашей базе знаний
2. **Показываем их модели** в промпте
3. **Модель учится на примерах** и генерирует лучше

## Почему RAG лучше дообучения?

| Критерий | Дообучение | RAG |
|----------|-----------|-----|
| **Скорость внедрения** | Недели | Часы |
| **Нужно примеров** | 500-1000+ | 10-50 |
| **Обновление базы** | Переобучить модель | Добавить строчку в JSON |
| **Стоимость** | GPU + время | Бесплатно (локально) |
| **Контроль** | Чёрный ящик | Видишь какие примеры используются |
| **Отладка** | Сложно | Легко (меняешь примеры) |

## Архитектура системы

```
┌──────────────────┐
│  Новая книга     │
│  (сюжет)         │
└────────┬─────────┘
         │
         ↓
┌────────────────────────────────┐
│  RAG: Поиск похожих примеров   │
│  (TF-IDF / embeddings)         │
└────────┬───────────────────────┘
         │
         ↓
┌────────────────────────────────┐
│  Промпт = Инструкция +         │
│           Похожие примеры +    │
│           Новый сюжет          │
└────────┬───────────────────────┘
         │
         ↓
┌────────────────────────────────┐
│  LLM генерирует заголовок      │
│  (учится на примерах)          │
└────────┬───────────────────────┘
         │
         ↓
┌────────────────────────────────┐
│  Оценка качества + выбор       │
│  лучшего из N попыток          │
└────────────────────────────────┘
```

## Как работает поиск похожих примеров

### Вариант 1: TF-IDF (простой, быстрый)

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Создаём векторы для всех примеров
vectorizer = TfidfVectorizer()
example_vectors = vectorizer.fit_transform([
    "студент убийство старуха топор",
    "офицер любовь измена поезд",
    "дворянин дуэль друг смерть"
])

# 2. Новый сюжет тоже в вектор
new_plot = "молодой человек убил богатую женщину"
new_vector = vectorizer.transform([new_plot])

# 3. Считаем похожесть (cosine similarity)
similarities = cosine_similarity(new_vector, example_vectors)
# → [0.85, 0.12, 0.23] — первый пример самый похожий!
```

**Плюсы:** Быстро, не требует GPU  
**Минусы:** Не понимает синонимы ("убийство" ≠ "преступление")

### Вариант 2: Embeddings (умный, медленнее)

```python
from sentence_transformers import SentenceTransformer

# 1. Загрузить модель для русского языка
model = SentenceTransformer('paraphrase-multilingual-mpnet-base-v2')

# 2. Закодировать примеры
example_embeddings = model.encode([
    "студент убил старуху",
    "офицер влюбился в замужнюю",
    "дворянин застрелил друга"
])

# 3. Новый сюжет
new_embedding = model.encode(["молодой человек совершил преступление"])

# 4. Похожесть
similarities = cosine_similarity(new_embedding, example_embeddings)
```

**Плюсы:** Понимает синонимы и смысл  
**Минусы:** Нужна отдельная модель (~500MB)

## Структура базы примеров

### Простой вариант (examples_db.json)

```json
{
  "examples": [
    {
      "book": "Преступление и наказание",
      "author": "Достоевский",
      "plot_keywords": ["студент", "убийство", "старуха", "совесть"],
      "headline": "Студент убил старуху-процентщицу: подробности",
      "score": 0.95
    }
  ]
}
```

### Расширенный вариант (с метаданными)

```json
{
  "examples": [
    {
      "book": "Преступление и наказание",
      "author": "Достоевский",
      "genre": "роман",
      "themes": ["преступление", "наказание", "совесть"],
      "characters": {
        "protagonist": "студент",
        "victim": "старуха"
      },
      "plot_keywords": ["студент", "убийство", "старуха", "топор"],
      "headline": "Студент убил старуху-процентщицу топором",
      "score": 0.95,
      "generated_at": "2025-03-09",
      "model": "qwen2.5:3b",
      "human_approved": true
    }
  ]
}
```

## Как пополнять базу примеров

### Автоматически

```python
# После генерации хорошего заголовка
if score >= 0.85:
    # Извлечь ключевые слова из сюжета
    keywords = extract_keywords(plot)
    
    # Добавить в базу
    rag.add_example(
        book=title,
        author=author,
        keywords=keywords,
        headline=headline,
        score=score
    )
```

### Вручную (курирование)

```python
# review_tool.py
def review_generated_headlines():
    """Интерактивный обзор сгенерированных заголовков"""
    
    for result in load_results():
        print(f"\nКнига: {result['title']}")
        print(f"Заголовок: {result['headline']}")
        print(f"Авто-оценка: {result['score']:.2f}")
        
        choice = input("Добавить в базу? (y/n/e=edit): ")
        
        if choice == 'y':
            rag.add_example(...)
        elif choice == 'e':
            edited = input("Новый заголовок: ")
            rag.add_example(..., headline=edited, score=1.0)
```

## Продвинутые техники

### 1. Few-Shot Learning (несколько примеров)

```python
# Показать модели 3-5 лучших примеров
similar = rag.find_similar(plot, top_k=5)

# В промпте:
"""
ПРИМЕРЫ:
1. [пример 1]
2. [пример 2]
3. [пример 3]

Теперь создай для: [новый сюжет]
"""
```

### 2. Negative Examples (контрпримеры)

```python
# Показать что НЕ надо делать
bad_examples = [
    "❌ Раскольников убил старуху (имя персонажа!)",
    "❌ В романе герой совершил преступление (мета-описание!)",
]

prompt = f"""
ХОРОШИЕ ПРИМЕРЫ:
{good_examples}

ПЛОХИЕ ПРИМЕРЫ (НЕ делай так):
{bad_examples}

Создай заголовок:
"""
```

### 3. Diversity Sampling (разнообразие)

```python
def find_diverse_examples(plot: str, top_k: int = 5):
    """Найти не только похожие, но и разнообразные примеры"""
    
    # 1. Найти 20 похожих
    candidates = rag.find_similar(plot, top_k=20)
    
    # 2. Выбрать 5 максимально непохожих друг на друга
    diverse = []
    diverse.append(candidates[0])  # Самый похожий
    
    for candidate in candidates[1:]:
        if len(diverse) >= top_k:
            break
        
        # Проверить насколько отличается от уже выбранных
        similarities = [
            compute_similarity(candidate, selected) 
            for selected in diverse
        ]
        
        if max(similarities) < 0.7:  # Достаточно отличается
            diverse.append(candidate)
    
    return diverse
```

### 4. Dynamic Prompting (адаптивный промпт)

```python
def build_adaptive_prompt(plot: str, rag: HeadlineRAG):
    """Адаптировать промпт под специфику сюжета"""
    
    # Определить жанр/тип сюжета
    if "убийство" in plot or "смерть" in plot:
        style = "криминальный"
        examples = rag.find_similar(plot, filter_by_theme="crime")
    
    elif "любовь" in plot or "измена" in plot:
        style = "романтический скандал"
        examples = rag.find_similar(plot, filter_by_theme="romance")
    
    else:
        style = "универсальный"
        examples = rag.find_similar(plot, top_k=3)
    
    prompt = f"""Стиль: {style}

Примеры для этого стиля:
{format_examples(examples)}

Создай заголовок:"""
    
    return prompt
```

## Метрики и A/B тестирование

### Отслеживание качества

```python
# tracking.json
{
  "experiments": [
    {
      "date": "2025-03-09",
      "method": "no_rag",
      "avg_score": 0.62,
      "high_quality_pct": 35
    },
    {
      "date": "2025-03-09",
      "method": "rag_tfidf_top3",
      "avg_score": 0.78,
      "high_quality_pct": 62
    },
    {
      "date": "2025-03-10",
      "method": "rag_embeddings_top5",
      "avg_score": 0.85,
      "high_quality_pct": 78
    }
  ]
}
```

### A/B тест

```python
def ab_test():
    """Сравнить RAG vs без RAG"""
    
    test_books = load_test_set(n=20)
    
    results_no_rag = []
    results_with_rag = []
    
    for book in test_books:
        # Без RAG
        headline_a = generate_baseline(book)
        score_a = score_headline(headline_a)
        results_no_rag.append(score_a)
        
        # С RAG
        headline_b = generate_with_rag(book, rag)
        score_b = score_headline(headline_b)
        results_with_rag.append(score_b)
    
    print(f"Без RAG: {np.mean(results_no_rag):.2f}")
    print(f"С RAG:   {np.mean(results_with_rag):.2f}")
    print(f"Улучшение: +{(np.mean(results_with_rag) - np.mean(results_no_rag)) * 100:.1f}%")
```

## Частые проблемы и решения

### Проблема 1: Модель копирует примеры слово в слово

**Решение:**
```python
# Добавить в промпт
"""
ВАЖНО: Не копируй примеры! Создай НОВЫЙ заголовок на основе сюжета.
Примеры показаны только для понимания СТИЛЯ.
"""

# Или проверить на плагиат
def is_plagiarism(generated: str, examples: List[str]) -> bool:
    for example in examples:
        similarity = compute_text_similarity(generated, example)
        if similarity > 0.8:  # Слишком похоже
            return True
    return False
```

### Проблема 2: Примеры не релевантные

**Решение:**
```python
# Фильтровать по минимальной похожести
similar = rag.find_similar(plot, top_k=10)
filtered = [ex for ex in similar if ex['similarity'] > 0.3]

if len(filtered) < 3:
    # Примеров мало — использовать generic промпт
    prompt = build_generic_prompt(plot)
else:
    prompt = build_rag_prompt(plot, filtered[:3])
```

### Проблема 3: База примеров разрастается

**Решение:**
```python
# Дедупликация: удалить очень похожие примеры
def deduplicate_examples(rag: HeadlineRAG, threshold: float = 0.9):
    """Оставить только уникальные примеры"""
    
    unique = []
    
    for example in rag.examples:
        # Проверить похожесть с уже добавленными
        is_duplicate = False
        
        for existing in unique:
            sim = compute_similarity(example['headline'], existing['headline'])
            if sim > threshold:
                # Дубликат! Оставить лучший
                if example['score'] > existing['score']:
                    unique.remove(existing)
                    unique.append(example)
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique.append(example)
    
    rag.examples = unique
    rag._build_index()
```

## Сравнение подходов

| Подход | Качество | Скорость | Сложность | Когда использовать |
|--------|----------|----------|-----------|-------------------|
| **Базовый промпт** | 60% | Быстро | Легко | Прототип |
| **Few-shot (3-5 примеров)** | 70% | Быстро | Легко | Статичные примеры |
| **RAG (TF-IDF)** | 80% | Средне | Средне | Динамический поиск |
| **RAG (Embeddings)** | 85% | Медленно | Средне | Лучшее качество |
| **Fine-tuning** | 90% | Быстро* | Сложно | Production с 1000+ примеров |

*после обучения

## Установка и запуск

```bash
# Установить зависимости
pip install scikit-learn requests --break-system-packages

# Для embeddings (опционально)
pip install sentence-transformers --break-system-packages

# Запустить
python generate_headlines_rag.py
```

## Дальнейшее развитие

1. **Гибридный поиск:** TF-IDF + Embeddings
2. **Мультимодальность:** Использовать обложки книг
3. **Пользовательская обратная связь:** Лайки/дизлайки
4. **Автоматическое A/B тестирование:** Непрерывное улучшение
5. **Экспорт в production:** API для массовой генерации

## Резюме

**RAG — идеальный компромисс между:**
- Простотой промпт-инжиниринга
- Качеством дообучения
- Гибкостью обновления

**Начни с RAG, если:**
- У тебя есть 10-50 хороших примеров
- Нужно быстро улучшить качество
- Важна возможность обновлять базу знаний
- Бюджет ограничен

**Переходи к дообучению, если:**
- У тебя 500+ примеров
- Генерируешь тысячи заголовков в день
- RAG даёт качество <85%
- Готов потратить время на инфраструктуру
