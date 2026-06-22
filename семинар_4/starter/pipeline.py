"""
pipeline.py — RAG над собственным корпусом (GPR/РЧ), сравнение стратегий чанкинга.
================================================================================
Адаптация каркаса семинара_4/starter под свой корпус. Отличия от стартера:
  • эмбеддер СМЕННЫЙ: на машине с интернетом — sentence-transformers
    (paraphrase-multilingual-MiniLM-L12-v2, как в стартере), а если модель не
    скачать — TF-IDF-фоллбэк (sklearn). Ретрив и eval работают БЕЗ LLM-ключа;
  • вместо ChromaDB — лёгкий in-memory косинусный индекс (тот же смысл, меньше
    зависимостей, запускается офлайн);
  • стратегия чанкинга — параметр: "fixed" (text[i:i+2000]) или
    "recursive" (RecursiveCharacterTextSplitter 400/80).

ЗАПУСК
------
  python pipeline.py ingest --strategy recursive
  python pipeline.py ask "Чем подавляют клаттер?" --strategy recursive
Шаг `ask` использует LLM (make_client, response_model=RAGAnswer). Без ключа он
падает в офлайн-режим и собирает экстрактивный ответ из найденных чанков —
ретрив при этом полноценный.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent / "data"

# ─────────────────────────── Чанкинг ───────────────────────────
_recursive = RecursiveCharacterTextSplitter(
    chunk_size=400, chunk_overlap=80, separators=["\n\n", "\n", ". ", "? ", "! ", " "]
)


def chunk_fixed(text: str, size: int = 2000) -> list[str]:
    """Стратегия A: рубим каждые N символов, без перекрытия."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def chunk_recursive(text: str) -> list[str]:
    """Стратегия B: рекурсивный сплиттер по абзацам/предложениям, overlap=80."""
    return [c.strip() for c in _recursive.split_text(text) if c.strip()]


CHUNKERS = {"fixed": chunk_fixed, "recursive": chunk_recursive}


def tokenize_ru(text: str) -> list[str]:
    return re.findall(r"[а-яa-z0-9ё+-]{2,}", text.lower())


# ─────────────────────────── Эмбеддер (сменный) ───────────────────────────
class DenseBackend:
    """Интерфейс: fit(docs) → запоминает матрицу документов; encode(queries)."""

    name = "base"

    def fit(self, docs: list[str]) -> None: ...
    def encode(self, queries: list[str]) -> np.ndarray: ...
    @property
    def doc_matrix(self) -> np.ndarray: ...


class STBackend(DenseBackend):
    """sentence-transformers (как в стартере). Требует скачивания модели."""

    name = "sentence-transformers"

    def __init__(self, model_name="paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self._docs = None

    def fit(self, docs):
        self._docs = self.model.encode(docs, normalize_embeddings=True)

    def encode(self, queries):
        return self.model.encode(queries, normalize_embeddings=True)

    @property
    def doc_matrix(self):
        return self._docs


class TfidfBackend(DenseBackend):
    """Офлайн-фоллбэк: лексические TF-IDF-векторы (sklearn). Без скачиваний."""

    name = "tfidf-fallback"

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.vec = TfidfVectorizer(
            tokenizer=tokenize_ru, token_pattern=None, ngram_range=(1, 2), min_df=1
        )
        self._docs = None

    def fit(self, docs):
        m = self.vec.fit_transform(docs).toarray()
        self._docs = _l2(m)

    def encode(self, queries):
        return _l2(self.vec.transform(queries).toarray())

    @property
    def doc_matrix(self):
        return self._docs


def _l2(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return m / n


def make_backend() -> DenseBackend:
    """sentence-transformers, если доступен и модель грузится; иначе TF-IDF."""
    try:
        b = STBackend()
        print(f"[dense] backend: {b.name}")
        return b
    except Exception as e:
        print(f"[dense] sentence-transformers недоступен ({type(e).__name__}); "
              f"использую TF-IDF-фоллбэк (ретрив остаётся рабочим).")
        return TfidfBackend()


# ─────────────────────────── Индекс ───────────────────────────
@dataclass
class RagIndex:
    strategy: str
    backend: DenseBackend
    ids: list[str] = field(default_factory=list)
    texts: list[str] = field(default_factory=list)
    bm25: BM25Okapi | None = None

    @classmethod
    def build(cls, strategy: str, backend: DenseBackend | None = None) -> "RagIndex":
        assert strategy in CHUNKERS, strategy
        backend = backend or make_backend()
        chunker = CHUNKERS[strategy]
        ids, texts = [], []
        for f in sorted(list(DATA_DIR.glob("*.txt")) + list(DATA_DIR.glob("*.md"))):
            if f.name == "gold.json":
                continue
            for i, c in enumerate(chunker(f.read_text(encoding="utf-8"))):
                ids.append(f"{f.stem}__{i}")
                texts.append(c)
        backend.fit(texts)
        bm25 = BM25Okapi([tokenize_ru(t) for t in texts])
        print(f"[ingest:{strategy}] {len(texts)} чанков из "
              f"{len(set(i.split('__')[0] for i in ids))} документов")
        return cls(strategy, backend, ids, texts, bm25)

    # — dense —
    def dense(self, query: str, k: int = 5) -> dict:
        q = self.backend.encode([query])[0]
        scores = self.backend.doc_matrix @ q
        order = np.argsort(scores)[::-1][:k]
        return self._pack(order)

    # — sparse —
    def sparse(self, query: str, k: int = 5) -> dict:
        scores = self.bm25.get_scores(tokenize_ru(query))
        order = np.argsort(scores)[::-1][:k]
        return self._pack(order)

    # — hybrid: dense + bm25 + RRF —
    def hybrid(self, query: str, k: int = 5, top: int = 15, c: int = 60) -> dict:
        q = self.backend.encode([query])[0]
        dense_order = np.argsort(self.backend.doc_matrix @ q)[::-1][:top]
        sparse_order = np.argsort(self.bm25.get_scores(tokenize_ru(query)))[::-1][:top]
        rrf: dict[int, float] = {}
        for rank, idx in enumerate(dense_order):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (c + rank)
        for rank, idx in enumerate(sparse_order):
            rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (c + rank)
        order = [i for i, _ in sorted(rrf.items(), key=lambda kv: kv[1], reverse=True)[:k]]
        return self._pack(order)

    def retrieve(self, query: str, k: int = 5, mode: str = "hybrid") -> dict:
        return {"dense": self.dense, "sparse": self.sparse, "hybrid": self.hybrid}[mode](query, k)

    def _pack(self, order) -> dict:
        ids = [self.ids[i] for i in order]
        docs = [self.texts[i] for i in order]
        return {"ids": [ids], "documents": [docs]}


# ─────────────────────────── Генерация (LLM или офлайн) ───────────────────────────
def build_prompt(query: str, hits: dict) -> str:
    ctx = "\n\n---\n\n".join(
        f"[{i}]\n{d}" for i, d in zip(hits["ids"][0], hits["documents"][0])
    )
    return (
        "Ты отвечаешь на технический вопрос по личной базе знаний об РЧ/георадарах. "
        "Опирайся ТОЛЬКО на контекст ниже; если ответа нет — скажи прямо.\n"
        "В quotes — 1-5 точных коротких цитат; в sources — id блоков; "
        "в confidence — честная оценка 0..1.\n\n"
        f"Контекст:\n{ctx}\n\nВопрос: {query}\n\nОтвет:"
    )


def ask(query: str, strategy: str = "recursive", k: int = 5):
    index = RagIndex.build(strategy)
    hits = index.retrieve(query, k=k, mode="hybrid")
    ids = hits["ids"][0]
    print(f"\nВОПРОС: {query}\nНайдено: {', '.join(ids)}")

    from schema import RAGAnswer

    try:
        from llm_client import get_model, make_client

        client = make_client()
        resp = client.chat.completions.create(
            model=get_model(),
            response_model=RAGAnswer,
            max_retries=3,
            temperature=0.2,
            messages=[{"role": "user", "content": build_prompt(query, hits)}],
        )
        print("\n[LLM]", resp.model_dump_json(indent=2))
        return resp
    except Exception as e:
        # офлайн: экстрактивный ответ из топ-чанка
        print(f"\n[offline] LLM недоступен ({type(e).__name__}). Экстрактивный ответ из контекста.")
        top = hits["documents"][0][0]
        sent = re.split(r"(?<=[.!?])\s+", top.strip())
        resp = RAGAnswer(
            answer=" ".join(sent[:2]),
            quotes=[sent[0][:160]] if sent else [],
            sources=ids[:3],
            confidence=0.5,
        )
        print(resp.model_dump_json(indent=2))
        return resp


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["ingest", "ask"])
    p.add_argument("query", nargs="?", default="")
    p.add_argument("--strategy", default="recursive", choices=list(CHUNKERS))
    p.add_argument("--k", type=int, default=5)
    a = p.parse_args()
    if a.cmd == "ingest":
        RagIndex.build(a.strategy)
    else:
        ask(a.query, a.strategy, a.k)
