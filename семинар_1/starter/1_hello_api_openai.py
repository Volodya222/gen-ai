"""
Семинар 1 — Часть 1: Первый запрос к API (OpenAI-совместимый)
==============================================================
Минимальный скрипт: отправляем один промпт, получаем ответ.

Клиент берём из llm_client.py — это обёртка над `openai.OpenAI`,
которая:
  • читает .env (LLM_BASE_URL / LLM_AUTH_TOKEN / LLM_MODEL, либо OPENAI_API_KEY);
  • пропускает проверку TLS на развернутых моделях (httpx.Client(verify=False));
  • отключает reasoning модели,
    чтобы ответ приходил быстро.

Интерфейс `.chat.completions.create(...)` — идентичен стандартному OpenAI SDK.
"""

from llm_client import get_model, make_raw_client

client = make_raw_client()
MODEL = get_model()

response = client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "Ты — помощник экономиста."},
        {"role": "user", "content": "Объясни в одном предложении, что такое инфляция."},
    ],
    temperature=0,
    max_tokens=200,
)

msg = response.choices[0].message.content
usage = response.usage

print("Ответ модели:")
print(msg)
print(
    f"\nТокены: вход={usage.prompt_tokens}, "
    f"выход={usage.completion_tokens}, всего={usage.total_tokens}"
)
