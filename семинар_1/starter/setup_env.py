"""
Семинар 1 — Часть 1: Проверка окружения
========================================
Запустите этот скрипт, чтобы убедиться, что всё установлено корректно.

Установка пакетов:
    pip install -r requirements.txt

Файл .env (создайте в той же папке, см. .env.example):

    # Вариант A: внутренний/self-hosted OpenAI-совместимый endpoint
    LLM_BASE_URL=https://inference.parsers360.ru:10443/v1
    LLM_AUTH_TOKEN=...
    LLM_MODEL=llm

    # Вариант B: публичный OpenAI
    # OPENAI_API_KEY=sk-...
    # LLM_MODEL=gpt-4.1-mini

    # Дополнительно — если собираешься прогонять gemini-версию:
    # GEMINI_API_KEY=AIza...
"""

import sys

def check():
    ok = True
    for pkg, name in [
        ("openai", "openai"),
        ("httpx", "httpx"),
        ("google.genai", "google-genai"),
        ("tiktoken", "tiktoken"),
        ("dotenv", "python-dotenv"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]:
        try:
            __import__(pkg)
            print(f"  ✓  {name}")
        except ImportError:
            print(f"  ✗  {name}  ← pip install {name}")
            ok = False

    # Проверяем .env
    from dotenv import load_dotenv
    import os
    load_dotenv()

    # Основная пара: self-hosted vs публичный OpenAI
    base = os.getenv("LLM_BASE_URL", "")
    auth = os.getenv("LLM_AUTH_TOKEN", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    if base and auth:
        model = os.getenv("LLM_MODEL", "(по умолчанию: llm)")
        print(f"  ✓  LLM_BASE_URL задан ({base})")
        print(f"  ✓  LLM_AUTH_TOKEN найден ({auth[:8]}...)")
        print(f"  ✓  LLM_MODEL = {model}")
    elif openai_key:
        model = os.getenv("LLM_MODEL", "gpt-4.1-mini")
        print(f"  ✓  OPENAI_API_KEY найден ({openai_key[:8]}...) — идём в публичный OpenAI")
        print(f"  ✓  LLM_MODEL = {model}")
    else:
        print("  ✗  Не задана ни пара LLM_BASE_URL+LLM_AUTH_TOKEN, ни OPENAI_API_KEY")
        print("     Открой .env и заполни один из вариантов (см. шапку этого файла).")
        ok = False

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key:
        print(f"  ✓  GEMINI_API_KEY найден ({gemini_key[:8]}...) — gemini-ветки доступны")
    else:
        print("  ⚠  GEMINI_API_KEY не задан (нужен только для *_gemini.py скриптов)")

    if ok:
        print("\nОкружение готово!")
    else:
        print("\nУстановите недостающие пакеты / заполните .env и запустите скрипт снова.")
    return ok

if __name__ == "__main__":
    print("Проверка окружения для Семинара 1\n")
    print(f"Python {sys.version}\n")
    check()
