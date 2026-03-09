"""
Инструмент для ручного review и корректировки заголовков
Позволяет:
- Оценить заголовки вручную
- Исправить заголовки
- Добавить лучшие в базу RAG
- Пропустить плохие
"""

import json
import re
from typing import List, Dict

# ========== КОНФИГУРАЦИЯ ==========

RESULTS_FILE = "yellow_headlines_rag.json"
EXAMPLES_DB = "examples_db.json"
REVIEWED_FILE = "yellow_headlines_reviewed.json"

# ========== ФУНКЦИИ ==========

def load_results(filename: str) -> List[Dict]:
    """Загрузить результаты генерации"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"❌ Файл {filename} не найден!")
        return []

def load_examples_db(filename: str) -> Dict:
    """Загрузить базу RAG примеров"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"examples": []}

def save_examples_db(data: Dict, filename: str):
    """Сохранить базу RAG примеров"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ База сохранена в {filename}")

def save_reviewed(results: List[Dict], filename: str):
    """Сохранить отрецензированные результаты"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"✅ Отрецензированные результаты сохранены в {filename}")

def extract_keywords(text: str, top_k: int = 10) -> List[str]:
    """Извлечь ключевые слова из текста"""
    # Простой метод: частотный анализ
    words = re.findall(r'\b[а-яё]{4,}\b', text.lower())
    
    # Убрать стоп-слова
    stop_words = {'быть', 'этот', 'весь', 'который', 'свой', 'один', 'мочь', 
                  'сказать', 'говорить', 'очень', 'хотеть', 'самый', 'знать'}
    
    words = [w for w in words if w not in stop_words]
    
    # Частоты
    from collections import Counter
    freq = Counter(words)
    
    # Топ-K
    return [word for word, _ in freq.most_common(top_k)]

def review_headlines():
    """Интерактивный review заголовков"""
    
    # Загрузить данные
    results = load_results(RESULTS_FILE)
    examples_db = load_examples_db(EXAMPLES_DB)
    
    if not results:
        print("Нет результатов для review!")
        return
    
    print("="*70)
    print("РУЧНОЙ REVIEW ЗАГОЛОВКОВ")
    print("="*70)
    print(f"\nВсего заголовков: {len(results)}")
    print(f"Примеров в базе RAG: {len(examples_db['examples'])}\n")
    
    print("КОМАНДЫ:")
    print("  [Enter] — оставить как есть")
    print("  [число 0-1] — изменить оценку (например: 0.95)")
    print("  [e] — отредактировать заголовок")
    print("  [+] — добавить в базу RAG (независимо от оценки)")
    print("  [s] — пропустить (не сохранять)")
    print("  [q] — закончить review\n")
    
    reviewed = []
    added_to_rag = 0
    skipped_count = 0
    
    for i, result in enumerate(results, 1):
        title = result['meta']['title']
        author = result['meta'].get('author', 'Неизвестен')
        headline = result['suggestions'][0]
        auto_score = result['meta']['score']
        
        print("\n" + "="*70)
        print(f"[{i}/{len(results)}] {title}")
        print(f"Автор: {author}")
        print("-"*70)
        print(f"Заголовок: {headline}")
        print(f"Авто-оценка: {auto_score:.2f}")
        print("-"*70)
        
        # Показать краткое содержание (первые 300 символов)
        input_text = result.get('input', '')
        plot_preview = input_text[:300] + "..." if len(input_text) > 300 else input_text
        print(f"\nКраткое содержание:\n{plot_preview}\n")
        
        choice = input("Действие: ").strip().lower()
        
        # Quit
        if choice == 'q':
            print("\n⏹️  Review прерван")
            break
        
        # Skip
        if choice == 's':
            print("⏭️  Пропущено")
            skipped_count += 1
            continue
        
        # Edit headline
        if choice == 'e':
            new_headline = input("Новый заголовок: ").strip()
            if new_headline:
                headline = new_headline
                result['suggestions'][0] = headline
                # Ручное редактирование = высокая оценка
                auto_score = 1.0
                result['meta']['score'] = 1.0
                result['meta']['manually_edited'] = True
                print(f"✅ Заголовок обновлён (оценка: 1.0)")
        
        # Change score
        elif choice and choice not in ['+', '']:
            try:
                new_score = float(choice)
                if 0 <= new_score <= 1:
                    auto_score = new_score
                    result['meta']['score'] = new_score
                    result['meta']['manually_scored'] = True
                    print(f"✅ Оценка изменена на {new_score:.2f}")
                else:
                    print("⚠️ Оценка должна быть от 0 до 1")
            except ValueError:
                print("⚠️ Неверный формат оценки")
        
        # Add to RAG
        if choice == '+' or auto_score >= 0.85:
            print("\n➕ Добавление в базу RAG...")
            
            # Извлечь ключевые слова
            keywords = extract_keywords(input_text, top_k=10)
            
            # Показать ключевые слова
            print(f"Автоматически извлечённые ключевые слова:")
            print(f"{', '.join(keywords)}")
            
            edit_keywords = input("Редактировать? (y/n): ").strip().lower()
            
            if edit_keywords == 'y':
                keywords_input = input("Введите ключевые слова через запятую: ").strip()
                if keywords_input:
                    keywords = [k.strip() for k in keywords_input.split(',')]
            
            # Добавить в базу
            new_example = {
                "book": title,
                "author": author,
                "plot_keywords": keywords,
                "headline": headline,
                "score": auto_score
            }
            
            examples_db['examples'].append(new_example)
            result['meta']['added_to_rag'] = True
            added_to_rag += 1
            
            print(f"✅ Добавлено в RAG! (всего примеров: {len(examples_db['examples'])})")
        
        # Сохранить результат
        reviewed.append(result)
    
    # Сохранить изменения
    if reviewed:
        save_reviewed(reviewed, REVIEWED_FILE)
        save_examples_db(examples_db, EXAMPLES_DB)
        
        print("\n" + "="*70)
        print("📊 ИТОГИ REVIEW:")
        print(f"   Отрецензировано: {len(reviewed)}")
        print(f"   Пропущено: {skipped_count}")
        print(f"   Добавлено в RAG: {added_to_rag}")
        print(f"   Всего примеров в RAG: {len(examples_db['examples'])}")
        print("="*70)
    else:
        print("\n⚠️ Нет изменений для сохранения")

# ========== ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ==========

def show_rag_database():
    """Показать текущую базу RAG"""
    
    examples_db = load_examples_db(EXAMPLES_DB)
    
    print("\n" + "="*70)
    print("БАЗА RAG ПРИМЕРОВ")
    print("="*70)
    print(f"\nВсего примеров: {len(examples_db['examples'])}\n")
    
    for i, ex in enumerate(examples_db['examples'], 1):
        print(f"{i}. {ex['book']} ({ex.get('author', 'Неизвестен')})")
        print(f"   → {ex['headline']}")
        print(f"   Оценка: {ex['score']:.2f}")
        print(f"   Ключевые слова: {', '.join(ex['plot_keywords'][:5])}...")
        print()

def remove_from_rag(index: int):
    """Удалить пример из базы RAG"""
    
    examples_db = load_examples_db(EXAMPLES_DB)
    
    if 0 <= index < len(examples_db['examples']):
        removed = examples_db['examples'].pop(index)
        save_examples_db(examples_db, EXAMPLES_DB)
        print(f"✅ Удалён: {removed['headline']}")
    else:
        print("❌ Неверный индекс")

def export_best_headlines(min_score: float = 0.80):
    """Экспортировать лучшие заголовки"""
    
    results = load_results(RESULTS_FILE)
    
    best = [r for r in results if r['meta']['score'] >= min_score]
    best.sort(key=lambda x: x['meta']['score'], reverse=True)
    
    filename = f"best_headlines_{min_score:.2f}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(f"ЛУЧШИЕ ЗАГОЛОВКИ (оценка ≥ {min_score})\n")
        f.write(f"Всего: {len(best)}\n\n")
        
        for i, r in enumerate(best, 1):
            f.write(f"{i}. {r['meta']['title']} (оценка: {r['meta']['score']:.2f})\n")
            f.write(f"   → {r['suggestions'][0]}\n\n")
    
    print(f"✅ Экспортировано {len(best)} заголовков в {filename}")

# ========== МЕНЮ ==========

def main_menu():
    """Главное меню"""
    
    while True:
        print("\n" + "="*70)
        print("УПРАВЛЕНИЕ ЗАГОЛОВКАМИ И RAG")
        print("="*70)
        print("\n1. Интерактивный review заголовков")
        print("2. Показать базу RAG примеров")
        print("3. Удалить пример из RAG")
        print("4. Экспортировать лучшие заголовки")
        print("5. Выход")
        
        choice = input("\nВыбери действие (1-5): ").strip()
        
        if choice == '1':
            review_headlines()
        
        elif choice == '2':
            show_rag_database()
        
        elif choice == '3':
            show_rag_database()
            try:
                idx = int(input("\nНомер примера для удаления (начиная с 1): ")) - 1
                remove_from_rag(idx)
            except ValueError:
                print("❌ Неверный номер")
        
        elif choice == '4':
            min_score = input("Минимальная оценка (по умолчанию 0.80): ").strip()
            min_score = float(min_score) if min_score else 0.80
            export_best_headlines(min_score)
        
        elif choice == '5':
            print("\n👋 До встречи!")
            break
        
        else:
            print("❌ Неверный выбор")

# ========== ЗАПУСК ==========

if __name__ == "__main__":
    main_menu()
