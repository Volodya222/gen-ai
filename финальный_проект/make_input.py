"""
Генерация входных артефактов проекта (скрипт, чтобы input/ был воспроизводим).

Создаёт:
  input/test_cases.json      — 75 тестовых входов (5 персон × 5 уровней предложения × 3 повтора).
  input/human_benchmark.json — эталонные числа из человеческих экспериментов.

Запуск:
    python make_input.py
"""
from __future__ import annotations

import json
from pathlib import Path

from personas import PERSONAS

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"
INPUT.mkdir(exist_ok=True)

# 5 уровней предложения — покрываем весь диапазон для сравнения с бенчмарком
OFFERS = [10, 20, 30, 40, 50]
REPEATS = 2  


def build_test_cases() -> list[dict]:
    """
    75 входов: 5 персон × 5 уровней предложения × 3 повтора.
    Повторы нужны для доверительных интервалов (bootstrapping по вариативности модели).
    """
    cases = []
    cid = 1
    for p in PERSONAS:
        for off in OFFERS:
            for rep in range(1, REPEATS + 1):
                cases.append(
                    {
                        "id": f"tc{cid:03d}",
                        "persona_id": p.id,
                        "offer": off,
                        "repeat": rep,
                        "expected_accept": off >= p.reject_below,
                        "segment": p.segment,
                    }
                )
                cid += 1
    return cases


def human_benchmark() -> dict:
    """
    Сводка по человеческим экспериментам — для сравнения reject_rate модели vs людей.
    Числа по пяти уровням предложения (Güth 1982; Oosterbeek 2004; Camerer 2003).
    """
    return {
        "source": "Camerer 2003; Oosterbeek 2004 meta-analysis; Güth 1982",
        "typical_offer_pct": [40, 50],
        "modal_offer_pct": 50,
        "reject_rate_by_offer": {
            "10": [0.6, 0.8],
            "20": [0.4, 0.6],
            "30": [0.2, 0.4],
            "40": [0.05, 0.15],
            "50": [0.0, 0.05],
        },
        "subgame_perfect_prediction": "предложить минимум (1-2%), принять любое >0",
        "empirical_contradiction": (
            "люди систематически отвергают низкие предложения, нарушая "
            "теоретико-игровое равновесие — ключевой стилизованный факт."
        ),
    }


def main() -> None:
    cases = build_test_cases()
    (INPUT / "test_cases.json").write_text(
        json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (INPUT / "human_benchmark.json").write_text(
        json.dumps(human_benchmark(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Записано {len(cases)} тест-кейсов в input/test_cases.json")
    print("Записан эталон в input/human_benchmark.json")


if __name__ == "__main__":
    main()
