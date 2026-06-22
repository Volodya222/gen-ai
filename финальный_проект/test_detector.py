"""
Мини-тест детектора галлюцинаций (без API).

Показывает, что детектор реально ловит ghost-раунды и выдуманные числа,
а не просто объявлен. Запуск:  python test_detector.py
"""
from __future__ import annotations

from pipeline import detect_hallucinations
from schema import ResponderMove


def main() -> None:
    # 1) ghost-раунд: в раунде 2 ссылка на раунд 5 (которого не было)
    m1 = ResponderMove(
        accept=False,
        fairness_perceived="unfair",
        reasoning="как в прошлый раз отказался",
        refers_to_round=5,
    )
    f1 = detect_hallucinations(m1, round_num=2, offer=10)
    assert any("ghost_round" in x for x in f1), f1

    # 2) выдуманное число: offer=30, а в обосновании всплыло 70 и 999→игнор, но 70 keep ок;
    #    возьмём 33 — не связано с раундом
    m2 = ResponderMove(
        accept=True,
        fairness_perceived="fair",
        reasoning="мне предложили 33 рубля, нормально",  # реально offer=30
        refers_to_round=None,
    )
    f2 = detect_hallucinations(m2, round_num=1, offer=30)
    assert any("ghost_number" in x for x in f2), f2

    # 3) чистый ход — без флагов
    m3 = ResponderMove(
        accept=True,
        fairness_perceived="fair",
        reasoning="предложили 45 из 100, это справедливо",
        refers_to_round=None,
    )
    f3 = detect_hallucinations(m3, round_num=1, offer=45)
    assert f3 == [], f3

    print("OK: детектор ловит ghost_round и ghost_number, чистый ход не флагует.")
    print(f"  кейс 1: {f1}")
    print(f"  кейс 2: {f2}")
    print(f"  кейс 3: {f3}")


if __name__ == "__main__":
    main()
