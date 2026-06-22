"""
eval.py — hit-rate@5 на gold-сете для двух стратегий чанкинга × двух режимов ретрива.
====================================================================================
Метрика (как в стартере): hit-rate@5 на уровне ДОКУМЕНТА-источника.
Для одного вопроса score = доля gold_sources, попавших в топ-5 чанков
(смотрим префикс id до '__'). Итог — среднее по всем вопросам.

Запуск:
    python eval.py            # все 4 конфигурации + сохранение output/results.json
"""

from __future__ import annotations

import json
from pathlib import Path

from pipeline import RagIndex, make_backend

GOLD = Path(__file__).parent / "data" / "gold.json"
OUT = Path(__file__).parent / "output"
OUT.mkdir(exist_ok=True)


def hit_rate(retrieved_ids: list[str], gold_sources: list[str]) -> float:
    retrieved = {rid.split("__")[0] for rid in retrieved_ids}
    found = [g for g in gold_sources if g in retrieved]
    return len(found) / len(gold_sources)


def run_config(index: RagIndex, gold: list[dict], mode: str, k: int = 5) -> dict:
    rows, total = [], 0.0
    for item in gold:
        hits = index.retrieve(item["question"], k=k, mode=mode)
        rids = hits["ids"][0]
        score = hit_rate(rids, item["gold_sources"])
        total += score
        rows.append({
            "id": item["id"], "type": item["type"], "score": round(score, 3),
            "gold": item["gold_sources"],
            "retrieved_sources": [r.split("__")[0] for r in rids],
        })
    return {"mean": round(total / len(gold), 3), "rows": rows}


def main():
    gold = json.loads(GOLD.read_text(encoding="utf-8"))
    backend = make_backend()  # один эмбеддер на все конфигурации

    # считаем для нескольких k: @5 — основная метрика ДЗ, @3/@1 — дискриминативные
    KS = [5, 3, 1]
    results = {}
    indexes = {}
    for strategy in ["fixed", "recursive"]:
        indexes[strategy] = RagIndex.build(strategy, backend=backend)
        for mode in ["dense", "hybrid"]:
            for k in KS:
                results[f"{strategy}/{mode}/k{k}"] = run_config(indexes[strategy], gold, mode, k=k)

    # ── Основная таблица: hit-rate@5 (требование ДЗ) ──
    print("\n" + "=" * 52)
    print("ОСНОВНАЯ МЕТРИКА — hit-rate@5")
    print(f"{'СТРАТЕГИЯ ЧАНКИНГА':<22}{'dense-only':>14}{'hybrid':>14}")
    print("-" * 52)
    for strategy in ["fixed", "recursive"]:
        d = results[f"{strategy}/dense/k5"]["mean"]
        h = results[f"{strategy}/hybrid/k5"]["mean"]
        label = "fixed (2000)" if strategy == "fixed" else "recursive (400/80)"
        print(f"{label:<22}{d:>14.2f}{h:>14.2f}")
    print("=" * 52)

    # ── Дискриминативная таблица: @5 / @3 / @1 (hybrid) ──
    print("\nДискриминативно (hybrid), hit-rate@k:")
    print(f"{'СТРАТЕГИЯ':<22}{'@5':>8}{'@3':>8}{'@1':>8}")
    for strategy in ["fixed", "recursive"]:
        row = [results[f"{strategy}/hybrid/k{k}"]["mean"] for k in KS]
        label = "fixed (2000)" if strategy == "fixed" else "recursive (400/80)"
        print(f"{label:<22}" + "".join(f"{v:>8.2f}" for v in row))

    # ── Где стратегии расходятся (на самом строгом информативном k=3) ──
    fx = {r["id"]: r for r in results["fixed/hybrid/k3"]["rows"]}
    rc = {r["id"]: r for r in results["recursive/hybrid/k3"]["rows"]}
    print("\nРасхождения по вопросам при @3 (hybrid):")
    for item in gold:
        i = item["id"]
        a, b = fx[i]["score"], rc[i]["score"]
        if a != b:
            w = "recursive" if b > a else "fixed"
            print(f"  #{i} ({item['type']}): fixed={a:.2f} vs recursive={b:.2f} → {w}")
            print(f"      gold={item['gold_sources']}")
            print(f"      fixed→{fx[i]['retrieved_sources']}")
            print(f"      recur→{rc[i]['retrieved_sources']}")

    (OUT / "results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nСохранено: {OUT/'results.json'}")
    return results


if __name__ == "__main__":
    main()
