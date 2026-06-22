"""
schema.py — Pydantic-схемы пайплайна анализа экспертных интервью (GPR/РЧ).
==========================================================================
Адаптация семинарской схемы фокус-группы под новую область:
  Participant  → Expert      (эксперт вместо участника)
  Concern      → Claim        (тезис вместо жалобы)
  аспекты price/speed/ux/... → новизна/обоснованность/практичность/риски

Требования ДЗ выполнены:
  • ≥1 Optional      — Expert.years_experience
  • ≥1 Literal       — ClaimCategory, AspectName, sentiment-поля, support
  • ≥1 field_validator — Claim.quote (непустая, осмысленной длины) +
                         Expert.years_experience (правдоподобный диапазон)
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

# Фиксированные справочники
ClaimCategory = Literal[
    "методология", "аппаратура", "обработка_сигналов", "ограничения", "применение"
]
AspectName = Literal["новизна", "обоснованность", "практичность", "риски"]
ALL_ASPECTS: list[str] = ["новизна", "обоснованность", "практичность", "риски"]


# ══════════════════════════════════════════════════════════
# Раунд 1 — Information Extraction
# ══════════════════════════════════════════════════════════
class Claim(BaseModel):
    category: ClaimCategory
    strength: int = Field(ge=1, le=5, description="сила/категоричность тезиса 1-5")
    quote: str

    @field_validator("quote")
    @classmethod
    def quote_must_be_meaningful(cls, v: str) -> str:
        # Бизнес-инвариант: цитата не пустая и не обрывок в пару символов.
        if len(v.strip()) < 10:
            raise ValueError("Цитата слишком короткая — вероятно, обрывок")
        return v.strip()


class Expert(BaseModel):
    name: str
    affiliation: str = ""
    years_experience: Optional[int] = None  # ← Optional (может быть не указан)
    specialization: str
    claims: list[Claim]
    related_methods: list[str] = Field(default_factory=list)

    @field_validator("years_experience")
    @classmethod
    def experience_plausible(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (0 <= v <= 60):
            raise ValueError("Стаж вне правдоподобного диапазона 0-60 лет")
        return v


# ══════════════════════════════════════════════════════════
# Раунд 2 — Аспектный анализ
# ══════════════════════════════════════════════════════════
class AspectSentiment(BaseModel):
    aspect: AspectName
    sentiment: Literal["positive", "negative", "neutral"]
    quote: str
    confidence: float = Field(ge=0, le=1)


class ExpertSentiment(BaseModel):
    name: str
    aspects: list[AspectSentiment]


# ══════════════════════════════════════════════════════════
# Раунд 2.5 — Autodiscovery аспектов (для «отлично»)
# ══════════════════════════════════════════════════════════
class DiscoveredAspect(BaseModel):
    name: str
    description: str = Field(min_length=5)


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect] = Field(min_length=3, max_length=12)


# ══════════════════════════════════════════════════════════
# Раунд 3 — Map-Reduce-резюме
# ══════════════════════════════════════════════════════════
class ChunkSummary(BaseModel):
    speaker: str
    key_points: list[str] = Field(min_length=1, max_length=6)
    sentiment: Literal["positive", "negative", "mixed"]


class DiscussionSummary(BaseModel):
    headline: str
    key_findings: list[str] = Field(min_length=2, max_length=8)
    action_items: list[str] = Field(min_length=1, max_length=8)


# ══════════════════════════════════════════════════════════
# Раунд 5 — LLM-as-judge
# ══════════════════════════════════════════════════════════
class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str] = Field(default_factory=list)
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float = Field(ge=0, le=1)
    summary: str


# ══════════════════════════════════════════════════════════
# Раунд 7 — Multi-doc свод (для «отлично»)
# ══════════════════════════════════════════════════════════
class MultiDocSummary(BaseModel):
    common_themes: list[str] = Field(min_length=1, max_length=8)
    unique_per_expert: dict[str, list[str]]
    overall_headline: str
