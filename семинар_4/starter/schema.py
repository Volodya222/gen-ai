"""schema.py — Pydantic-схема ответа RAG."""

from __future__ import annotations

from pydantic import BaseModel, Field


class RAGAnswer(BaseModel):
    answer: str
    quotes: list[str] = Field(default_factory=list, description="1-5 точных коротких цитат")
    sources: list[str] = Field(default_factory=list, description="id чанков-источников")
    confidence: float = Field(ge=0, le=1)
