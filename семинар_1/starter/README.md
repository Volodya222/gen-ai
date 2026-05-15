# Семинар 1 — стартер

Тут лежит «скелет» — рабочее окружение и шесть скриптов в правильном порядке.
Часть из них уже готовая, часть — пустые места с пометкой `← ВПИШИ СЮДА`. Это
твоя работа на семинаре: дополнить эти места и увидеть, как меняется поведение
модели.

## Установка (5 минут)

```bash
# 1. Виртуальное окружение
python -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# 2. Зависимости
pip install -r requirements.txt
либо
uv add --requirements  "семинар_1/starter/requirements.txt"


# 3. Скопируй .env.example в .env и впиши свой LLM_AUTH_TOKEN
cp .env.example .env
# затем открой .env в редакторе и впиши токен

# 4. Проверка, что всё работает
python setup_env.py
```


## Что делать, если застрял

- Перечитай комментарии вокруг TODO — там обычно есть подсказка.
- Задай вопрос.
- После пары в `../solution/` появится эталонная версия — сравни со своей.

## Что НЕ делать

- Не запускать `prompt_experiment.py` со всеми пустыми SYSTEM до того,
  как впишешь хоть один — будет неинтересно.
- Не коммитить `.env` в git (он в `.gitignore`).
- Не закрывать ноутбук на перерыве, пока не сохранил `headlines_scored.csv`.

## LLM-эндпоинт

Используем DeepSeek V4 Flash. Токен выдаётся преподавателем.

Альтернативы (если есть свои ключи):

- публичный OpenAI — поставь `OPENAI_API_KEY` и `LLM_MODEL=gpt-4o-mini`
- Google Gemini — поставь `GEMINI_API_KEY` и используй `hello_api_gemini.py`

Подробности — в `starter/.env.example`.

## Типичные грабли

- `OpenAIError: api_key must be set` — `.env` не найден. Положи в `starter/`.
