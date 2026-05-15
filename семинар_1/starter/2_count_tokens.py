"""
Семинар 1 — Часть 2: Подсчёт токенов через tiktoken
=====================================================
tiktoken — библиотека OpenAI для подсчёта токенов без обращения к API.
У DeepSeek и Gemini свои токенизаторы, но tiktoken даёт хорошую оценку
порядка величины (±10–15%).

Что делаем:
  1. Сравниваем русский и английский — одна и та же фраза, сколько токенов.
  2. Считаем стоимость скоринга 20 заголовков в трёх моделях.
  3. ЗАДАНИЕ: посчитать стоимость вашего собственного текста (внизу скрипта).
"""

import tiktoken

# Загружаем токенизатор модели gpt-4o-mini.
# Для DeepSeek V4 Flash точного токенизатора в tiktoken нет — берём cl100k_base
# как приближение. Для грубых прикидок этого достаточно.
enc = tiktoken.encoding_for_model("gpt-4o-mini")


def count(text: str) -> int:
    """Сколько токенов в строке (по cl100k_base токенизатору)."""
    return len(enc.encode(text))


def cost(
    text: str, n_calls: int, price_in: float, price_out: float, out_tokens: int = 5
) -> float:
    """
    Стоимость в долларах: text прогоняем n_calls раз, на выходе out_tokens.
    Цены — за миллион токенов.
    """
    in_tokens = count(text)
    return n_calls * (in_tokens * price_in + out_tokens * price_out) / 1_000_000


# ───────────────────────────────────────────────────────────────────────
# Прайс-лист моделей (за 1M токенов, USD)
# Цифры на май 2026; для актуальных — обращаться к документации.
# ───────────────────────────────────────────────────────────────────────
PRICES = {
    # модель              → (input $/1M,  output $/1M)
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o": (2.50, 10.00),
    "deepseek-v4-flash": (0.14, 0.28),
    "deepseek-v4-pro": (1.74, 3.48),
    "claude-haiku-4.5": (1.00, 5.00),
}


# ─── Пример 1: английский vs русский ───

text_en = "The market is efficient"
text_ru = "Рынок является эффективным"

print("═" * 60)
print("Сравнение: английский vs русский")
print("═" * 60)
print(f'EN: "{text_en}"')
print(f"    Токенов: {count(text_en)}")
print(f"    Токены:  {enc.encode(text_en)}")
print()
print(f'RU: "{text_ru}"')
print(f"    Токенов: {count(text_ru)}")
print(f"    Токены:  {enc.encode(text_ru)}")
print(f"\n    Русский дороже в {count(text_ru) / count(text_en):.1f} раз")
print("\n    → правило большого пальца: 1 русский символ ≈ 1 токен,")
print("      1 английское слово ≈ 1.3 токена.")


# ─── Пример 2: расчёт стоимости эксперимента ───

print("\n" + "═" * 60)
print("Сколько стоит прогнать 20 заголовков, 5 раз каждый")
print("═" * 60)

system_prompt = (
    "Ты — эксперт по медиа и журналистике. "
    "Оцени кликбейтность заголовка по шкале от 1 до 10. "
    "Ответь только числом."
)
sample_headline = "ВЫ НЕ ПОВЕРИТЕ, что произошло с рублём за одну ночь!"
full_prompt = system_prompt + "\n\nЗаголовок: " + sample_headline

n_headlines = 20
n_runs = 5
total_calls = n_headlines * n_runs

print(f"\nЗапросов всего: {n_headlines} × {n_runs} = {total_calls}")
print(f"Промпт на запрос: {count(full_prompt)} токенов")
print(f"Ответ на запрос: ~5 токенов\n")

print(f"{'Модель':<24}{'Цена эксперимента':<24}{'Цена 1М таких':<20}")
print("─" * 60)
for model, (p_in, p_out) in PRICES.items():
    exp_cost = cost(full_prompt, total_calls, p_in, p_out, out_tokens=5)
    million_cost = cost(full_prompt, 1_000_000, p_in, p_out, out_tokens=5)
    exp_str = f"${exp_cost:.4f}"
    if exp_cost < 0.01:
        exp_str += f" ({exp_cost * 100:.2f} ¢)"
    print(f"{model:<24}{exp_str:<24}${million_cost:.2f}")


# ─── Задание для Вас ───

print("\n" + "═" * 60)
print("ЗАДАНИЕ: ваш собственный текст")
print("═" * 60)

# ─── ВСТАВЬТЕ СВОЙ ТЕКСТ СЮДА ───
# Это может быть абзац курсовой, аннотация статьи, описание продукта.
# Чем больше, тем интереснее.

my_text = """
Заменить эту тройную строку текстом из своей курсовой или статьи.
Скрипт посчитает токены и стоимость прогона через каждую модель.
""".strip()

# ─── ────────────────────────── ───

my_tokens = count(my_text)
print(f"\nДлина текста: {len(my_text)} символов, {my_tokens} токенов")
print(f"({my_tokens / max(len(my_text.split()), 1):.2f} токена на слово)")

print(f"\nСтоимость прогнать этот текст 100 раз (с коротким ответом):")
print("─" * 60)
for model, (p_in, p_out) in PRICES.items():
    c = cost(my_text, 100, p_in, p_out, out_tokens=50)
    print(f"  {model:<24} ${c:.4f}" + (f"   ({c * 100:.2f} ¢)" if c < 0.10 else ""))

print(f"\nА если ответ модели тоже на весь объём текста (~{my_tokens} токенов)?")
print("─" * 60)
for model, (p_in, p_out) in PRICES.items():
    c = cost(my_text, 100, p_in, p_out, out_tokens=my_tokens)
    print(f"  {model:<24} ${c:.4f}" + (f"   ({c * 100:.2f} ¢)" if c < 0.10 else ""))
