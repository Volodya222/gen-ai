"""
Генератор синтетических заявок на курсы повышения квалификации (ДПО).
=====================================================================

Что делает:
  1. Стратифицированно (КВОТА: ровно 5 заявок на каждый из 10 городов = 50)
     запрашивает у LLM по одной заявке за раз, передавая `seed_city` в промпт —
     это и борьба с mode collapse, и гарантия ровного распределения по городам.
  2. Каждая заявка валидируется схемой `Application` через make_client(
     response_model=Application, max_retries=3): невалидный ответ модель
     переделывает автоматически (до 3 попыток).
  3. Результат сохраняется в applications.csv (вложенный address распакован
     в колонки city/district) + строятся гистограммы cities.png и specialities.png.

ПЕРЕЗАПУСК С РАБОЧИМ КЛЮЧОМ
--------------------------
Чтобы сгенерировать данные «по-настоящему» через LLM:
  1. cp .env.example .env  и впишите LLM_BASE_URL / LLM_AUTH_TOKEN / LLM_MODEL.
  2. python generator.py
Скрипт сам пойдёт в эндпоинт.

ОФЛАЙН-РЕЖИМ (без ключа)
------------------------
Если ключ/эндпоинт недоступен (как сейчас — преподаватель не оплатил токены),
make_client() не сможет подключиться. Тогда скрипт берёт встроенный датасет
FALLBACK_APPLICATIONS (50 заявок, сгенерированных заранее) и прогоняет его
через ТУ ЖЕ валидацию + CSV + графики. Пайплайн идентичен — меняется только
источник сырых данных. Никаких правок кода для переключения не нужно.
"""

import json
import random
import sys
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import ValidationError

from schema import Application

# Стратификация: список городов и квота на каждый.
CITIES_ORDERED = [
    "Москва", "Санкт-Петербург", "Новосибирск", "Екатеринбург", "Казань",
    "Нижний Новгород", "Самара", "Краснодар", "Челябинск", "Ростов-на-Дону",
]
QUOTA_PER_CITY = 5  # 10 городов × 5 = 50 заявок

SYSTEM_PROMPT = (
    "Ты генерируешь синтетические заявки слушателей на курсы повышения "
    "квалификации (ДПО) в России. Каждая заявка — правдоподобный человек: "
    "ФИО, возраст (22-65), адрес (город и район), текущая специальность, "
    "желаемый курс, стаж работы и год окончания вуза. Год окончания и возраст "
    "не должны противоречить друг другу."
)


def build_user_prompt(seed_city: str) -> str:
    """Промпт под одну заявку. seed_city фиксирует город (стратификация),
    nonce заставляет модель разнообразить остальные поля (анти-collapse)."""
    nonce = random.randint(1000, 9999)
    return (
        f"Сгенерируй ОДНУ заявку. Город проживания строго: {seed_city}. "
        f"Сделай человека непохожим на остальных — варьируй пол, возраст, район, "
        f"специальность и курс. seed={nonce}"
    )


# ───────────────────────────── РЕАЛЬНАЯ ГЕНЕРАЦИЯ ─────────────────────────────
def generate_via_llm() -> list[Application] | None:
    """Пытается сгенерировать 50 заявок через LLM. Возвращает None, если
    эндпоинт/ключ недоступен (тогда вызывающий код уходит в офлайн)."""
    try:
        from llm_client import get_model, make_client

        client = make_client()
        model = get_model()
    except Exception as e:  # нет ключа / нет llm_client / нет сети
        print(f"[offline] LLM недоступен ({type(e).__name__}: {e}).")
        return None

    apps: list[Application] = []
    for city in CITIES_ORDERED:
        for slot in range(QUOTA_PER_CITY):
            try:
                app = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": build_user_prompt(city)},
                    ],
                    response_model=Application,
                    max_retries=3,
                    temperature=0.9,
                )
                # На случай, если модель проигнорировала seed_city — чиним адрес.
                if app.address.city != city:
                    app.address.city = city
                apps.append(app)
                print(f"  [{len(apps):2d}/50] {city:18s} ← {app.full_name}")
            except Exception as e:
                print(f"  ✗ {city}: {type(e).__name__}: {e}")
    if len(apps) < 50:
        print(f"[offline] получено лишь {len(apps)}/50 — переключаюсь на fallback.")
        return None
    return apps


# ───────────────────────────── ОФЛАЙН-ИСТОЧНИК ─────────────────────────────
def generate_offline() -> list[Application]:
    """Берёт встроенный датасет и валидирует его той же схемой."""
    apps, invalid = [], 0
    for raw in FALLBACK_APPLICATIONS:
        try:
            apps.append(Application(**raw))
        except ValidationError as e:
            invalid += 1
            print(f"  ✗ невалидная запись: {e.errors()[0]['msg']}")
    print(f"[offline] валидных заявок: {len(apps)}/{len(FALLBACK_APPLICATIONS)}")
    return apps


# ───────────────────────────── СОХРАНЕНИЕ + ГРАФИКИ ─────────────────────────────
def to_dataframe(apps: list[Application]) -> pd.DataFrame:
    rows = []
    for a in apps:
        d = a.model_dump()
        addr = d.pop("address")
        d["city"] = addr["city"]
        d["district"] = addr["district"]
        rows.append(d)
    cols = ["full_name", "age", "city", "district", "speciality",
            "desired_course", "years_of_experience", "graduation_year"]
    return pd.DataFrame(rows)[cols]


def plot_bar(series: pd.Series, title: str, out: str, threshold_pct: float, color: str):
    counts = series.value_counts()
    n = counts.sum()
    plt.figure(figsize=(10, 4.5))
    bars = counts.plot.bar(color=color, edgecolor="white")
    plt.axhline(n * threshold_pct / 100, color="#D9534F", ls="--", lw=1,
                label=f"порог {threshold_pct:.0f}%")
    plt.title(f"{title} (n={n}, топ={counts.iloc[0] / n * 100:.0f}%)")
    plt.ylabel("Число заявок")
    plt.xticks(rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  → {out}")
    return counts


def main():
    print("=== Генерация заявок ДПО ===")
    apps = generate_via_llm()
    source = "LLM"
    if apps is None:
        apps = generate_offline()
        source = "offline-fallback"

    if not apps:
        sys.exit("Не удалось получить ни одной заявки.")

    df = to_dataframe(apps)
    df.to_csv("applications.csv", index=False, encoding="utf-8-sig")
    print(f"\n[{source}] сохранено {len(df)} заявок → applications.csv")

    print("\nГистограммы:")
    city_counts = plot_bar(df["city"], "Распределение по городам", "cities.png", 40, "#4A90D9")
    spec_counts = plot_bar(df["speciality"], "Распределение по специальностям",
                           "specialities.png", 35, "#5CB85C")

    # Короткая сводка в консоль
    print("\n── Сводка ──")
    print(f"Города: топ {city_counts.iloc[0] / len(df) * 100:.0f}% "
          f"({'OK' if city_counts.iloc[0] / len(df) <= 0.40 else 'ПРЕВЫШЕН'} порог 40%)")
    print(f"Специальности: топ {spec_counts.iloc[0] / len(df) * 100:.0f}% "
          f"({'OK' if spec_counts.iloc[0] / len(df) <= 0.35 else 'ПРЕВЫШЕН'} порог 35%)")
    print(f"Уникальных имён: {df['full_name'].nunique()}/{len(df)}")


# 50 заранее сгенерированных заявок — офлайн-источник (см. docstring выше).
FALLBACK_APPLICATIONS = [
    {
        "full_name": "Голубева Ксения Олеговна",
        "age": 39,
        "address": {
            "city": "Москва",
            "district": "Тверской"
        },
        "speciality": "Бухгалтер",
        "desired_course": "Кибербезопасность",
        "years_of_experience": 15,
        "graduation_year": 2009
    },
    {
        "full_name": "Михайлов Артём Сергеевич",
        "age": 35,
        "address": {
            "city": "Москва",
            "district": "Хамовники"
        },
        "speciality": "Бухгалтер",
        "desired_course": "Бухгалтерский учёт и налогообложение",
        "years_of_experience": 10,
        "graduation_year": 2016
    },
    {
        "full_name": "Михайлов Дмитрий Николаевич",
        "age": 24,
        "address": {
            "city": "Москва",
            "district": "Басманный"
        },
        "speciality": "IT-специалист",
        "desired_course": "Управление персоналом",
        "years_of_experience": 0,
        "graduation_year": 2024
    },
    {
        "full_name": "Зайцев Александр Александрович",
        "age": 43,
        "address": {
            "city": "Москва",
            "district": "Пресненский"
        },
        "speciality": "Учитель",
        "desired_course": "Управление проектами",
        "years_of_experience": 19,
        "graduation_year": 2005
    },
    {
        "full_name": "Михайлова Дарья Александровна",
        "age": 58,
        "address": {
            "city": "Москва",
            "district": "Замоскворечье"
        },
        "speciality": "Инженер-строитель",
        "desired_course": "Аналитика данных",
        "years_of_experience": 35,
        "graduation_year": 1991
    },
    {
        "full_name": "Кузнецова Мария Дмитриевна",
        "age": 52,
        "address": {
            "city": "Санкт-Петербург",
            "district": "Центральный"
        },
        "speciality": "HR-специалист",
        "desired_course": "Бухгалтерский учёт и налогообложение",
        "years_of_experience": 28,
        "graduation_year": 1997
    },
    {
        "full_name": "Кузнецова Татьяна Викторовна",
        "age": 33,
        "address": {
            "city": "Санкт-Петербург",
            "district": "Адмиралтейский"
        },
        "speciality": "HR-специалист",
        "desired_course": "Кибербезопасность",
        "years_of_experience": 9,
        "graduation_year": 2014
    },
    {
        "full_name": "Зайцева Анна Михайловна",
        "age": 35,
        "address": {
            "city": "Санкт-Петербург",
            "district": "Василеостровский"
        },
        "speciality": "Инженер-строитель",
        "desired_course": "Веб-разработка",
        "years_of_experience": 4,
        "graduation_year": 2019
    },
    {
        "full_name": "Петрова Дарья Викторовна",
        "age": 33,
        "address": {
            "city": "Санкт-Петербург",
            "district": "Петроградский"
        },
        "speciality": "Маркетолог",
        "desired_course": "Управление персоналом",
        "years_of_experience": 2,
        "graduation_year": 2021
    },
    {
        "full_name": "Михайлов Егор Сергеевич",
        "age": 31,
        "address": {
            "city": "Санкт-Петербург",
            "district": "Московский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Кибербезопасность",
        "years_of_experience": 2,
        "graduation_year": 2024
    },
    {
        "full_name": "Андреева Юлия Михайловна",
        "age": 24,
        "address": {
            "city": "Новосибирск",
            "district": "Центральный"
        },
        "speciality": "HR-специалист",
        "desired_course": "Веб-разработка",
        "years_of_experience": 0,
        "graduation_year": 2024
    },
    {
        "full_name": "Виноградов Михаил Андреевич",
        "age": 33,
        "address": {
            "city": "Новосибирск",
            "district": "Ленинский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Управление проектами",
        "years_of_experience": 2,
        "graduation_year": 2024
    },
    {
        "full_name": "Соколова Ольга Андреевна",
        "age": 39,
        "address": {
            "city": "Новосибирск",
            "district": "Октябрьский"
        },
        "speciality": "Врач",
        "desired_course": "Аналитика данных",
        "years_of_experience": 8,
        "graduation_year": 2015
    },
    {
        "full_name": "Андреев Михаил Сергеевич",
        "age": 27,
        "address": {
            "city": "Новосибирск",
            "district": "Заельцовский"
        },
        "speciality": "Логист",
        "desired_course": "Охрана труда",
        "years_of_experience": 1,
        "graduation_year": 2023
    },
    {
        "full_name": "Зайцев Николай Дмитриевич",
        "age": 37,
        "address": {
            "city": "Новосибирск",
            "district": "Дзержинский"
        },
        "speciality": "Маркетолог",
        "desired_course": "Управление персоналом",
        "years_of_experience": 2,
        "graduation_year": 2021
    },
    {
        "full_name": "Семёнов Роман Николаевич",
        "age": 26,
        "address": {
            "city": "Екатеринбург",
            "district": "Ленинский"
        },
        "speciality": "IT-специалист",
        "desired_course": "Веб-разработка",
        "years_of_experience": 1,
        "graduation_year": 2024
    },
    {
        "full_name": "Андреева Мария Николаевна",
        "age": 44,
        "address": {
            "city": "Екатеринбург",
            "district": "Октябрьский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Управление проектами",
        "years_of_experience": 16,
        "graduation_year": 2010
    },
    {
        "full_name": "Белова Ксения Олеговна",
        "age": 24,
        "address": {
            "city": "Екатеринбург",
            "district": "Верх-Исетский"
        },
        "speciality": "Маркетолог",
        "desired_course": "Кибербезопасность",
        "years_of_experience": 1,
        "graduation_year": 2024
    },
    {
        "full_name": "Зайцев Павел Иванович",
        "age": 29,
        "address": {
            "city": "Екатеринбург",
            "district": "Чкаловский"
        },
        "speciality": "Инженер-строитель",
        "desired_course": "Охрана труда",
        "years_of_experience": 2,
        "graduation_year": 2024
    },
    {
        "full_name": "Зайцева Татьяна Михайловна",
        "age": 41,
        "address": {
            "city": "Екатеринбург",
            "district": "Кировский"
        },
        "speciality": "Юрист",
        "desired_course": "Аналитика данных",
        "years_of_experience": 13,
        "graduation_year": 2010
    },
    {
        "full_name": "Голубев Александр Николаевич",
        "age": 33,
        "address": {
            "city": "Казань",
            "district": "Вахитовский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Кибербезопасность",
        "years_of_experience": 11,
        "graduation_year": 2015
    },
    {
        "full_name": "Богданова Елена Александровна",
        "age": 31,
        "address": {
            "city": "Казань",
            "district": "Ново-Савиновский"
        },
        "speciality": "Бухгалтер",
        "desired_course": "Веб-разработка",
        "years_of_experience": 5,
        "graduation_year": 2019
    },
    {
        "full_name": "Андреева Мария Олеговна",
        "age": 29,
        "address": {
            "city": "Казань",
            "district": "Приволжский"
        },
        "speciality": "Юрист",
        "desired_course": "Охрана труда",
        "years_of_experience": 5,
        "graduation_year": 2019
    },
    {
        "full_name": "Воробьёва Светлана Николаевна",
        "age": 27,
        "address": {
            "city": "Казань",
            "district": "Советский"
        },
        "speciality": "IT-специалист",
        "desired_course": "Аналитика данных",
        "years_of_experience": 5,
        "graduation_year": 2020
    },
    {
        "full_name": "Лебедева Татьяна Дмитриевна",
        "age": 43,
        "address": {
            "city": "Казань",
            "district": "Московский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Охрана труда",
        "years_of_experience": 19,
        "graduation_year": 2006
    },
    {
        "full_name": "Голубев Тимур Николаевич",
        "age": 43,
        "address": {
            "city": "Нижний Новгород",
            "district": "Нижегородский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Цифровой маркетинг",
        "years_of_experience": 22,
        "graduation_year": 2004
    },
    {
        "full_name": "Андреева Юлия Петровна",
        "age": 35,
        "address": {
            "city": "Нижний Новгород",
            "district": "Советский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Веб-разработка",
        "years_of_experience": 13,
        "graduation_year": 2013
    },
    {
        "full_name": "Морозова Мария Михайловна",
        "age": 27,
        "address": {
            "city": "Нижний Новгород",
            "district": "Канавинский"
        },
        "speciality": "IT-специалист",
        "desired_course": "Аналитика данных",
        "years_of_experience": 1,
        "graduation_year": 2023
    },
    {
        "full_name": "Андреева Светлана Ивановна",
        "age": 23,
        "address": {
            "city": "Нижний Новгород",
            "district": "Автозаводский"
        },
        "speciality": "IT-специалист",
        "desired_course": "Аналитика данных",
        "years_of_experience": 2,
        "graduation_year": 2024
    },
    {
        "full_name": "Беляев Сергей Петрович",
        "age": 41,
        "address": {
            "city": "Нижний Новгород",
            "district": "Приокский"
        },
        "speciality": "IT-специалист",
        "desired_course": "Управление персоналом",
        "years_of_experience": 3,
        "graduation_year": 2021
    },
    {
        "full_name": "Голубева Виктория Андреевна",
        "age": 29,
        "address": {
            "city": "Самара",
            "district": "Ленинский"
        },
        "speciality": "Маркетолог",
        "desired_course": "Цифровой маркетинг",
        "years_of_experience": 4,
        "graduation_year": 2019
    },
    {
        "full_name": "Морозова Елена Ивановна",
        "age": 46,
        "address": {
            "city": "Самара",
            "district": "Самарский"
        },
        "speciality": "Инженер-строитель",
        "desired_course": "Бухгалтерский учёт и налогообложение",
        "years_of_experience": 10,
        "graduation_year": 2016
    },
    {
        "full_name": "Петров Павел Дмитриевич",
        "age": 64,
        "address": {
            "city": "Самара",
            "district": "Октябрьский"
        },
        "speciality": "Учитель",
        "desired_course": "Цифровой маркетинг",
        "years_of_experience": 29,
        "graduation_year": 1994
    },
    {
        "full_name": "Воробьёва Виктория Ивановна",
        "age": 35,
        "address": {
            "city": "Самара",
            "district": "Промышленный"
        },
        "speciality": "Логист",
        "desired_course": "Цифровой маркетинг",
        "years_of_experience": 12,
        "graduation_year": 2012
    },
    {
        "full_name": "Богданов Егор Михайлович",
        "age": 27,
        "address": {
            "city": "Самара",
            "district": "Кировский"
        },
        "speciality": "IT-специалист",
        "desired_course": "Аналитика данных",
        "years_of_experience": 5,
        "graduation_year": 2021
    },
    {
        "full_name": "Соколов Дмитрий Михайлович",
        "age": 49,
        "address": {
            "city": "Краснодар",
            "district": "Центральный"
        },
        "speciality": "Бухгалтер",
        "desired_course": "Управление проектами",
        "years_of_experience": 27,
        "graduation_year": 1999
    },
    {
        "full_name": "Виноградова Виктория Александровна",
        "age": 41,
        "address": {
            "city": "Краснодар",
            "district": "Западный"
        },
        "speciality": "Логист",
        "desired_course": "Цифровой маркетинг",
        "years_of_experience": 5,
        "graduation_year": 2021
    },
    {
        "full_name": "Фёдорова Ирина Дмитриевна",
        "age": 37,
        "address": {
            "city": "Краснодар",
            "district": "Карасунский"
        },
        "speciality": "Врач",
        "desired_course": "Кибербезопасность",
        "years_of_experience": 16,
        "graduation_year": 2010
    },
    {
        "full_name": "Беляева Ольга Михайловна",
        "age": 41,
        "address": {
            "city": "Краснодар",
            "district": "Прикубанский"
        },
        "speciality": "IT-специалист",
        "desired_course": "Бухгалтерский учёт и налогообложение",
        "years_of_experience": 19,
        "graduation_year": 2007
    },
    {
        "full_name": "Павлов Михаил Викторович",
        "age": 24,
        "address": {
            "city": "Краснодар",
            "district": "Центральный"
        },
        "speciality": "Маркетолог",
        "desired_course": "Цифровой маркетинг",
        "years_of_experience": 0,
        "graduation_year": 2024
    },
    {
        "full_name": "Петров Максим Викторович",
        "age": 55,
        "address": {
            "city": "Челябинск",
            "district": "Центральный"
        },
        "speciality": "Юрист",
        "desired_course": "Цифровой маркетинг",
        "years_of_experience": 31,
        "graduation_year": 1993
    },
    {
        "full_name": "Фёдоров Максим Иванович",
        "age": 31,
        "address": {
            "city": "Челябинск",
            "district": "Калининский"
        },
        "speciality": "HR-специалист",
        "desired_course": "Управление персоналом",
        "years_of_experience": 3,
        "graduation_year": 2023
    },
    {
        "full_name": "Виноградов Дмитрий Николаевич",
        "age": 31,
        "address": {
            "city": "Челябинск",
            "district": "Курчатовский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Аналитика данных",
        "years_of_experience": 9,
        "graduation_year": 2017
    },
    {
        "full_name": "Петров Сергей Андреевич",
        "age": 39,
        "address": {
            "city": "Челябинск",
            "district": "Ленинский"
        },
        "speciality": "Бухгалтер",
        "desired_course": "Управление проектами",
        "years_of_experience": 13,
        "graduation_year": 2012
    },
    {
        "full_name": "Павлова Мария Сергеевна",
        "age": 37,
        "address": {
            "city": "Челябинск",
            "district": "Советский"
        },
        "speciality": "Учитель",
        "desired_course": "Аналитика данных",
        "years_of_experience": 15,
        "graduation_year": 2011
    },
    {
        "full_name": "Тарасов Роман Олегович",
        "age": 37,
        "address": {
            "city": "Ростов-на-Дону",
            "district": "Кировский"
        },
        "speciality": "Юрист",
        "desired_course": "Веб-разработка",
        "years_of_experience": 15,
        "graduation_year": 2011
    },
    {
        "full_name": "Соколов Тимур Сергеевич",
        "age": 43,
        "address": {
            "city": "Ростов-на-Дону",
            "district": "Ленинский"
        },
        "speciality": "Менеджер по продажам",
        "desired_course": "Веб-разработка",
        "years_of_experience": 11,
        "graduation_year": 2015
    },
    {
        "full_name": "Богданов Максим Александрович",
        "age": 61,
        "address": {
            "city": "Ростов-на-Дону",
            "district": "Октябрьский"
        },
        "speciality": "Юрист",
        "desired_course": "Цифровой маркетинг",
        "years_of_experience": 36,
        "graduation_year": 1990
    },
    {
        "full_name": "Беляева Елена Петровна",
        "age": 29,
        "address": {
            "city": "Ростов-на-Дону",
            "district": "Пролетарский"
        },
        "speciality": "Бухгалтер",
        "desired_course": "Управление проектами",
        "years_of_experience": 0,
        "graduation_year": 2024
    },
    {
        "full_name": "Тарасов Тимур Петрович",
        "age": 39,
        "address": {
            "city": "Ростов-на-Дону",
            "district": "Ворошиловский"
        },
        "speciality": "Логист",
        "desired_course": "Управление проектами",
        "years_of_experience": 17,
        "graduation_year": 2009
    }
]


if __name__ == "__main__":
    main()
