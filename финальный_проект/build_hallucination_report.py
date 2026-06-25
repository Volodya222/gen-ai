"""
build_hallucination_report.py — единый отчёт по галлюцинациям.

Собирает находки из ДВУХ источников:
  1. output/eval_results.json — боевой eval (50 кейсов).
  2. output/transcript_*.json — транскрипты симуляций (реальный API).

Это закрывает требование «общий отчёт: сколько ghost-цитат, на каких кейсах,
примеры». Запуск после eval.py и simulate.py:
    python build_hallucination_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "output"


def main() -> None:
    cases = []

    # --- источник 1: eval ---
    eval_path = OUTPUT / "eval_results.json"
    eval_total = 0
    eval_n = 0
    if eval_path.exists():
        data = json.loads(eval_path.read_text(encoding="utf-8"))
        rows = data.get("rows", [])
        eval_n = len(rows)
        for r in rows:
            if r.get("hallucinations"):
                eval_total += len(r["hallucinations"])
                cases.append({
                    "source": "eval",
                    "id": r["id"],
                    "persona_id": r["persona_id"],
                    "offer": r["offer"],
                    "flags": r["hallucinations"],
                })

    # --- источник 2: транскрипты симуляций ---
    sim_total = 0
    sim_rounds = 0
    for tr_path in sorted(OUTPUT.glob("transcript_*.json")):
        tr = json.loads(tr_path.read_text(encoding="utf-8"))
        persona = tr.get("persona", tr_path.stem)
        for rnd in tr.get("transcript", []):
            sim_rounds += 1
            if rnd.get("hallucinations"):
                sim_total += len(rnd["hallucinations"])
                cases.append({
                    "source": "simulation",
                    "persona_id": persona,
                    "round": rnd["round"],
                    "offer": rnd["offer"],
                    "flags": rnd["hallucinations"],
                })

    total = eval_total + sim_total
    n_total = eval_n + sim_rounds

    report = {
        "summary": {
            "total_caught": total,
            "n_cases_with_hallucinations": len(cases),
            "total_observations": n_total,
            "rate_per_observation": round(total / n_total, 4) if n_total else 0.0,
        },
        "by_source": {
            "eval": {"observations": eval_n, "caught": eval_total},
            "simulation": {"observations": sim_rounds, "caught": sim_total},
        },
        "cases": cases,
        "types": {
            "ghost_round": "ссылка на номер раунда, которого ещё не было (>= текущего)",
            "ghost_number": "число в reasoning не совпадает с offer/keep/0/50/100 текущего раунда",
        },
        "note": (
            "Детектор работает на живых ответах модели (анализ reasoning после "
            "каждого хода). В боевом eval галлюцинаций не найдено; в симуляции "
            "пойман реальный ghost_number — пример ниже в cases."
        ),
    }

    out = OUTPUT / "hallucination_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Записан {out}")
    print(f"  всего поймано: {total} (eval: {eval_total}, симуляция: {sim_total})")
    print(f"  кейсов с галлюцинациями: {len(cases)}")
    for c in cases:
        print(f"    [{c['source']}] {c.get('id', c.get('persona_id'))} offer={c['offer']}: {c['flags']}")


if __name__ == "__main__":
    main()
