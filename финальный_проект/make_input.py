"""
Генерация входных артефактов проекта (скрипт, чтобы input/ был воспроизводим).

Создаёт:
  input/test_cases.json   — 15 тестовых входов (персона × предложение).
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


def build_test_cases() -> list[dict]:
    """
    15 входов: для каждой из 5 персон — 3 уровня предложения (низкое/среднее/
    почти-равное). Ожидаемое решение задаётся ground-truth порогом персоны:
    accept, если offer >= reject_below.
    """
    offers = [10, 30, 45]
    cases = []
    cid = 1
    for p in PERSONAS:
        for off in offers:
            cases.append(
                {
                    "id": f"tc{cid:02d}",
                    "persona_id": p.id,
                    "offer": off,
                    # ground-truth: примет ли человек такого типа (для exact-match)
                    "expected_accept": off >= p.reject_below,
                    "segment": p.segment,
                }
            )
            cid += 1
    return cases


def human_benchmark() -> dict:
    """
    Сводка по человеческим экспериментам с «Ультиматумом» — для проверки
    гипотезы трека A. Числа — устоявшиеся в литературе диапазоны
    (Güth 1982; Oosterbeek 2004 мета-анализ; Camerer 2003 «Behavioral Game Theory»).
    Это НЕ выдача синтетики за людей: это опубликованный человеческий эталон,
    с которым мы сравниваем поведение модели.
    """
    return {
        "source": "Camerer 2003; Oosterbeek 2004 meta-analysis; Güth 1982",
        "typical_offer_pct": [40, 50],
        "modal_offer_pct": 50,
        "low_offer_reject_rate": {
            "offer_le_20pct": [0.4, 0.6],   # доля отвержений предложений <=20%
            "offer_around_30pct": [0.2, 0.4],
            "offer_50pct": [0.0, 0.05],
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
