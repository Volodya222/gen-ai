"""
pipeline.py — конвейер анализа экспертных интервью (GPR/РЧ).
============================================================
analyze(input_path) собирает полный конвейер:
    IE  →  аспекты (+ check_quotes)  →  Map-Reduce  →  judge
плюс для критерия «отлично»:  multi-doc консолидация  +  autodiscovery аспектов.

ЗАПУСК С РАБОЧИМ КЛЮЧОМ
----------------------
  1. Положить файлы рядом с llm_client.py (семинар_3/starter) ИЛИ скопировать
     llm_client.py сюда.
  2. cp .env.example .env  и вписать LLM_BASE_URL / LLM_AUTH_TOKEN / LLM_MODEL.
  3. python pipeline.py
Тогда каждый этап реально ходит в модель: make_client(response_model=..., max_retries=3).

ОФЛАЙН-РЕЖИМ (без ключа)
------------------------
Если llm_client/ключ недоступны, конвейер берёт заранее подготовленные ответы из
fallback_data.py и прогоняет их через ТУ ЖЕ схему, check_quotes, heatmap и judge.
Меняется только источник «ответа модели» — структура и проверки идентичны.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from prompts import (
    ASPECTS_SYSTEM,
    DISCOVER_SYSTEM,
    IE_SYSTEM,
    JUDGE_SYSTEM,
    MULTI_DOC_SYSTEM,
    CHUNK_SYSTEM,
    REDUCE_SYSTEM,
)
from schema import (
    ALL_ASPECTS,
    ChunkSummary,
    DiscoveredAspects,
    DiscussionSummary,
    Expert,
    ExpertSentiment,
    JudgeReport,
    MultiDocSummary,
)

# Примерный прайс DeepSeek (USD за 1M токенов) — для оценки стоимости прогона.
PRICE_IN_PER_M = 0.27
PRICE_OUT_PER_M = 1.10

# ── Определяем режим: есть рабочий клиент или офлайн ──
try:
    from llm_client import get_model, make_client

    _client = make_client()
    MODEL = get_model()
    OFFLINE = False
except Exception as e:  # нет ключа / нет llm_client / нет сети
    print(f"[offline] LLM недоступен ({type(e).__name__}: {e}). Беру fallback_data.")
    import fallback_data as fb

    _client = None
    MODEL = "offline"
    OFFLINE = True

_usage = {"in": 0, "out": 0}


def _accrue(resp) -> None:
    u = getattr(resp, "usage", None)
    if u:
        _usage["in"] += getattr(u, "prompt_tokens", 0) or 0
        _usage["out"] += getattr(u, "completion_tokens", 0) or 0


def _call(response_model, system: str, user: str):
    """Один вызов модели со structured output и подсчётом usage."""
    obj, resp = _client.chat.completions.create(
        model=MODEL,
        response_model=response_model,
        max_retries=3,
        temperature=0.0,
        with_completion=True,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    _accrue(resp)
    return obj


# ─────────────────────────── Этапы ───────────────────────────
def load_corpus(input_path: str) -> list[tuple[str, str]]:
    """Вернуть [(имя_файла, текст), ...] для всех .txt в папке (или одного файла)."""
    p = Path(input_path)
    files = sorted(p.glob("*.txt")) if p.is_dir() else [p]
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def extract_experts(corpus: list[tuple[str, str]]) -> list[Expert]:
    if OFFLINE:
        return [Expert(**e) for e in fb.EXPERTS]
    out = []
    for _, text in corpus:
        out.append(_call(Expert, IE_SYSTEM, text))  # один эксперт на файл
    return out


def extract_aspects(corpus: list[tuple[str, str]]) -> list[ExpertSentiment]:
    if OFFLINE:
        return [ExpertSentiment(**a) for a in fb.ASPECTS]
    out = []
    for _, text in corpus:
        out.append(_call(ExpertSentiment, ASPECTS_SYSTEM, text))
    return out


def check_quotes(aspects: list[ExpertSentiment], full_text: str) -> list[tuple[str, str]]:
    """Ghost-цитаты: те, чьи первые 30 символов не находятся в исходном тексте."""
    t = full_text.lower()
    ghosts = []
    for p in aspects:
        for a in p.aspects:
            probe = a.quote.strip().lower()[:30]
            if probe and probe not in t:
                ghosts.append((p.name, a.quote))
    return ghosts


def build_heatmap(aspects: list[ExpertSentiment], out_path: str) -> None:
    names = [p.name.split()[0] for p in aspects]  # фамилии для компактности
    sent = {"positive": 1, "negative": -1, "neutral": 0}
    m = np.full((len(names), len(ALL_ASPECTS)), np.nan)
    for i, p in enumerate(aspects):
        for a in p.aspects:
            if a.aspect in ALL_ASPECTS:
                m[i, ALL_ASPECTS.index(a.aspect)] = sent[a.sentiment]
    plt.figure(figsize=(8, 4.5))
    sns.heatmap(m, annot=True, fmt=".0f", xticklabels=ALL_ASPECTS, yticklabels=names,
                center=0, cmap="RdYlGn", cbar_kws={"label": "sentiment (-1..+1)"})
    plt.title("Аспектная тональность по экспертам")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def map_reduce(corpus: list[tuple[str, str]]) -> tuple[list[ChunkSummary], DiscussionSummary]:
    if OFFLINE:
        chunks = [ChunkSummary(**c) for c in fb.CHUNK_SUMMARIES]
        return chunks, DiscussionSummary(**fb.SUMMARY)

    # MAP в параллель: один файл = один фрагмент
    texts = [t for _, t in corpus]
    summaries: list[ChunkSummary | None] = [None] * len(texts)
    with ThreadPoolExecutor(max_workers=5) as pool:
        futs = {pool.submit(_call, ChunkSummary, CHUNK_SYSTEM, t): i for i, t in enumerate(texts)}
        for fut in as_completed(futs):
            summaries[futs[fut]] = fut.result()
    chunks = [s for s in summaries if s is not None]

    # REDUCE
    joined = "\n\n".join(
        f"## {s.speaker} ({s.sentiment})\n" + "\n".join(f"- {p}" for p in s.key_points)
        for s in chunks
    )
    summary = _call(DiscussionSummary, REDUCE_SYSTEM, joined)
    return chunks, summary


def run_judge(experts: list[Expert], summary: DiscussionSummary) -> JudgeReport:
    if OFFLINE:
        return JudgeReport(**fb.JUDGE)
    parts = ["## Рекомендации (которые оцениваем)"]
    for i, a in enumerate(summary.action_items, 1):
        parts.append(f"  {i}. {a}")
    parts.append("\n## Тезисы экспертов (исходные данные)")
    for e in experts:
        for c in e.claims:
            parts.append(f"  - [{e.name}/{c.category}, strength={c.strength}] «{c.quote}»")
    return _call(JudgeReport, JUDGE_SYSTEM, "\n".join(parts))


def discover_aspects(full_text: str) -> DiscoveredAspects:
    if OFFLINE:
        return DiscoveredAspects(**fb.DISCOVERED)
    return _call(DiscoveredAspects, DISCOVER_SYSTEM, full_text)


def consolidate(chunks: list[ChunkSummary]) -> MultiDocSummary:
    if OFFLINE:
        return MultiDocSummary(**fb.MULTIDOC)
    joined = "\n\n".join(
        f"## {s.speaker}\n" + "\n".join(f"- {p}" for p in s.key_points) for s in chunks
    )
    return _call(MultiDocSummary, MULTI_DOC_SYSTEM, joined)


# ─────────────────────────── Оркестрация ───────────────────────────
def analyze(input_path: str, out_dir: str = "output") -> dict:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    corpus = load_corpus(input_path)
    full_text = "\n".join(t for _, t in corpus)
    t0 = time.time()
    print(f"[{'offline' if OFFLINE else 'LLM:' + MODEL}] источников: {len(corpus)}")

    # 1) IE
    experts = extract_experts(corpus)
    n_claims = sum(len(e.claims) for e in experts)
    (out / "experts.json").write_text(
        json.dumps([e.model_dump() for e in experts], ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(f"  IE: {len(experts)} экспертов, {n_claims} тезисов")

    # 2) Аспекты + check_quotes + heatmap
    aspects = extract_aspects(corpus)
    n_aspect = sum(len(a.aspects) for a in aspects)
    ghosts = check_quotes(aspects, full_text)
    build_heatmap(aspects, str(out / "heatmap.png"))
    (out / "aspects.json").write_text(
        json.dumps([a.model_dump() for a in aspects], ensure_ascii=False, indent=2),
        encoding="utf-8")
    pct = len(ghosts) / n_aspect * 100 if n_aspect else 0
    print(f"  Аспекты: {n_aspect} оценок | ghost-цитат: {len(ghosts)} ({pct:.0f}%)")
    for name, q in ghosts:
        print(f"    ⚠ ghost [{name}]: «{q[:70]}…»")

    # 3) Map-Reduce
    chunks, summary = map_reduce(corpus)
    (out / "summary.json").write_text(summary.model_dump_json(indent=2), encoding="utf-8")
    print(f"  Map-Reduce: {len(chunks)} мини-резюме → свод из {len(summary.action_items)} рекомендаций")

    # 4) Judge
    report = run_judge(experts, summary)
    (out / "judge_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
    cnt = {"supported": 0, "weakly_supported": 0, "not_supported": 0}
    for v in report.verdicts:
        cnt[v.support] += 1
    print(f"  Judge: overall_score={report.overall_score:.2f} | {cnt}")

    # 5) «Отлично»: autodiscovery + multi-doc
    discovered = discover_aspects(full_text)
    (out / "discovered_aspects.json").write_text(
        discovered.model_dump_json(indent=2), encoding="utf-8")
    fixed = set(ALL_ASPECTS)
    novel = [a.name for a in discovered.aspects if a.name not in fixed]
    print(f"  Autodiscovery: {len(discovered.aspects)} аспектов, новых вне Literal: {novel}")

    multidoc = consolidate(chunks)
    (out / "multidoc.json").write_text(multidoc.model_dump_json(indent=2), encoding="utf-8")
    print(f"  Multi-doc: общих тем {len(multidoc.common_themes)}, экспертов {len(multidoc.unique_per_expert)}")

    # Стоимость и время
    cost = _usage["in"] / 1e6 * PRICE_IN_PER_M + _usage["out"] / 1e6 * PRICE_OUT_PER_M
    elapsed = time.time() - t0
    cost_str = "$0.00 (офлайн)" if OFFLINE else f"${cost:.4f} ({_usage['in']}+{_usage['out']} ток.)"
    print(f"\n  Время: {elapsed:.1f}с | Стоимость: {cost_str}")
    if report.overall_score < 0.7:
        print("  ⚠ overall_score < 0.7 — стоит переписать REDUCE-промпт и прогнать заново.")

    return {
        "experts": len(experts), "claims": n_claims, "aspects": n_aspect,
        "ghosts": len(ghosts), "ghost_pct": pct, "overall_score": report.overall_score,
        "elapsed_s": elapsed, "cost_usd": 0.0 if OFFLINE else cost,
        "novel_aspects": novel,
    }


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "input"
    analyze(path)
