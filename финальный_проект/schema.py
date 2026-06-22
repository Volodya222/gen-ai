"""
Структурированные схемы проекта «Ультиматум: homo silicus».

Здесь живут все pydantic-модели, которыми мы фиксируем формат ответов LLM
(response_model в llm_client) и бизнес-инварианты предметной области через
field_validator.

Бизнес-инварианты игры «Ультиматум»:
  * Предлагающий делит ровно 100 ₽ → доля Отвечающему в [0, 100], целое.
  * Вердикт судьи о рациональности — оценка 1..5 (как «оценка 1–5» в ТЗ).
  * Любая ссылка на номер прошлого раунда не может быть больше текущего
    (ловим галлюцинации «вспомнил несуществующий раунд»).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Сколько денег делим в каждом раунде. Бизнес-константа предметной области.
PIE = 100


# ===========================================================================
# Ходы игроков
# ===========================================================================


class ProposerMove(BaseModel):
    """Ход Предлагающего: сколько из 100 ₽ отдать Отвечающему."""

    share_to_responder: int = Field(
        ...,
        description="Сколько из 100 ₽ предложить второму игроку (целое 0..100).",
    )
    reasoning: str = Field(
        default="", description="Одна короткая фраза, почему столько."
    )
    refers_to_round: Optional[int] = Field(
        default=None,
        description="Если ссылаешься на конкретный прошлый раунд — его номер, иначе null.",
    )

    @field_validator("share_to_responder")
    @classmethod
    def _share_in_pie(cls, v: int) -> int:
        # Бизнес-инвариант: делёж 100 ₽, доля вне [0,100] невозможна.
        if not 0 <= v <= PIE:
            raise ValueError(f"share_to_responder={v} вне диапазона 0..{PIE}")
        return v


class ResponderMove(BaseModel):
    """Ход Отвечающего: принять делёж или отвергнуть."""

    accept: bool = Field(..., description="Принять предложенный делёж или отвергнуть.")
    fairness_perceived: Literal["fair", "unfair", "borderline"] = Field(
        ...,
        description="Как воспринимается предложение: справедливо/несправедливо/на грани.",
    )
    reasoning: str = Field(default="", description="Одна короткая фраза, почему.")
    refers_to_round: Optional[int] = Field(
        default=None,
        description="Если ссылаешься на конкретный прошлый раунд — его номер, иначе null.",
    )


# ===========================================================================
# Вердикт судьи (LLM-as-judge)
# ===========================================================================


class JudgeVerdict(BaseModel):
    """
    Вердикт модели-судьи о ходе Отвечающего.

    Судья отвечает на вопрос: согласуется ли решение принять/отвергнуть с
    заявленным восприятием справедливости и с «человеческой» нормой
    (несправедливое часто отвергают из принципа)?
    """

    rationality_score: int = Field(
        ...,
        description="Насколько ход внутренне согласован и человекоподобен: 1 (бессвязно) .. 5 (образцово).",
    )
    consistent: bool = Field(
        ..., description="Согласуется ли решение с заявленным восприятием справедливости."
    )
    note: str = Field(default="", description="Одна фраза-обоснование вердикта.")

    @field_validator("rationality_score")
    @classmethod
    def _score_1_5(cls, v: int) -> int:
        # Бизнес-инвариант «оценка 1–5» прямо из ТЗ.
        if not 1 <= v <= 5:
            raise ValueError(f"rationality_score={v} вне диапазона 1..5")
        return v


# ===========================================================================
# Результат одного раунда (агрегируется пайплайном, не приходит от LLM целиком)
# ===========================================================================


class RoundResult(BaseModel):
    round: int
    persona_id: str
    offer: int
    accept: bool
    fairness_perceived: str
    proposer_payoff: int
    responder_payoff: int
    judge_score: int
    judge_consistent: bool

    @field_validator("offer")
    @classmethod
    def _offer_in_pie(cls, v: int) -> int:
        if not 0 <= v <= PIE:
            raise ValueError(f"offer={v} вне диапазона 0..{PIE}")
        return v
