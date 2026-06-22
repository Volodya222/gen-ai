"""
Pydantic-схема для заявки на курс повышения квалификации (ДПО).
===============================================================

Идея та же, что в семинаре с покупательскими персонами, но домен другой:
вместо «покупателя e-commerce» мы описываем «заявку слушателя на курс ДПО».

Что здесь происходит и зачем (на пальцах):
  * `Address` — ВЛОЖЕННАЯ модель (город + район). Pydantic валидирует
    вложенные объекты рекурсивно — модели часто спотыкаются именно на них.
  * `Application` — сама заявка. Поля с жёсткими границами (`Field(ge=..., le=...)`)
    и закрытыми списками (`Literal[...]`) — это и есть «фильтр», который
    ловит мусор от LLM и заставляет её переделать (см. make_client + max_retries).
  * Валидаторы:
      1. field_validator на `city`          — город обязан быть из CITIES;
      2. field_validator на `graduation_year`— год не позже текущего (и не до 1970);
      3. model_validator (кросс-полевой)     — возраст и год выпуска не должны
         противоречить друг другу (нельзя окончить вуз раньше ~20 лет или
         «в будущем»), а стаж не может превышать ни возраст, ни время с выпуска.

Файл импортируется и `generator.py` (генерация), и любой утилитой проверки.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# Текущий год берём динамически — тогда правило «год выпуска не позже текущего»
# остаётся верным и в следующем году без правок кода.
CURRENT_YEAR = datetime.now().year

# ─────────────────────────── Справочники ───────────────────────────
# Минимум 10 городов (требование ДЗ). Используем set для O(1)-проверки.
CITIES: set[str] = {
    "Москва",
    "Санкт-Петербург",
    "Новосибирск",
    "Екатеринбург",
    "Казань",
    "Нижний Новгород",
    "Самара",
    "Краснодар",
    "Челябинск",
    "Ростов-на-Дону",
}

# Минимум 8 специальностей.
Speciality = Literal[
    "Бухгалтер",
    "Учитель",
    "Инженер-строитель",
    "Врач",
    "Менеджер по продажам",
    "Юрист",
    "IT-специалист",
    "HR-специалист",
    "Маркетолог",
    "Логист",
]

# Минимум 6 желаемых курсов.
DesiredCourse = Literal[
    "Управление проектами",
    "Аналитика данных",
    "Цифровой маркетинг",
    "Кибербезопасность",
    "Бухгалтерский учёт и налогообложение",
    "Управление персоналом",
    "Веб-разработка",
    "Охрана труда",
]


# ─────────────────────────── Вложенная модель ───────────────────────────
class Address(BaseModel):
    """Адрес слушателя: город (из справочника) + район/округ."""

    city: str
    district: str = Field(min_length=2, max_length=40)

    @field_validator("city")
    @classmethod
    def city_must_be_in_list(cls, v: str) -> str:
        if v not in CITIES:
            raise ValueError(f"Город «{v}» не из утверждённого списка")
        return v


# ─────────────────────────── Основная модель ───────────────────────────
class Application(BaseModel):
    """Одна заявка на курс повышения квалификации."""

    full_name: str = Field(min_length=5, max_length=80)
    age: int = Field(ge=22, le=65)
    address: Address  # ← вложенный объект, а не плоский city: str
    speciality: Speciality
    desired_course: DesiredCourse
    years_of_experience: int = Field(ge=0, le=40)
    graduation_year: int = Field(ge=1980, le=2024)

    # Shortcut: application.city работает и для анализа, и для распаковки в CSV,
    # не заставляя везде писать application.address.city.
    @property
    def city(self) -> str:
        return self.address.city

    # ── Валидатор №2: год выпуска должен быть «человеческим» ──
    @field_validator("graduation_year")
    @classmethod
    def graduation_year_sane(cls, v: int) -> int:
        if v < 1970:
            raise ValueError("Год окончания вуза не может быть раньше 1970")
        if v > CURRENT_YEAR:
            raise ValueError(f"Год окончания вуза не может быть позже {CURRENT_YEAR}")
        return v

    # ── Валидатор №3: кросс-полевая согласованность возраст ↔ выпуск ↔ стаж ──
    @model_validator(mode="after")
    def consistency(self) -> "Application":
        birth_year = CURRENT_YEAR - self.age
        age_at_graduation = self.graduation_year - birth_year

        # Нельзя окончить вуз раньше ~20 лет и позже ~60 (поздняя переподготовка ок).
        if age_at_graduation < 20:
            raise ValueError(
                f"Противоречие: при возрасте {self.age} год выпуска {self.graduation_year} "
                f"означает выпуск в {age_at_graduation} лет (раньше 20 — нереалистично)"
            )
        if age_at_graduation > 60:
            raise ValueError(
                f"Противоречие: выпуск в {age_at_graduation} лет (позже 60 — нереалистично)"
            )

        # Стаж не может превышать время с момента выпуска (+2 года на подработку в вузе).
        years_since_grad = CURRENT_YEAR - self.graduation_year
        if self.years_of_experience > years_since_grad + 2:
            raise ValueError(
                f"Противоречие: стаж {self.years_of_experience} лет больше, чем прошло "
                f"с выпуска ({years_since_grad} лет)"
            )

        # И тривиально: работать с 16 лет, не раньше.
        if self.years_of_experience > self.age - 16:
            raise ValueError(
                f"Противоречие: стаж {self.years_of_experience} при возрасте {self.age}"
            )

        return self
