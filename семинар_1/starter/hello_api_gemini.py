"""
Семинар 1 — Часть 1: Первый запрос к API (Google Gemini)
=========================================================
Минимальный скрипт: отправляем один промпт, получаем ответ.

Документация: https://ai.google.dev/gemini-api/docs
"""

import os

from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-3-flash-preview",
    contents="Объясни в одном предложении, что такое инфляция.",
    config={
        "system_instruction": "Ты — помощник экономиста.",
        "temperature": 0,
        "max_output_tokens": 200,
    },
)

print("Ответ модели:")
print(response.text)
print(
    f"\nТокены: вход={response.usage_metadata.prompt_token_count}, "
    f"выход={response.usage_metadata.candidates_token_count}, "
    f"всего={response.usage_metadata.total_token_count}"
)
