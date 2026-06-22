"""
Инструменты агента (техника курса: агент с инструментами / tool use).

Предлагающий — это не «одна болталка», а агент, который ПЕРЕД ходом обязан
посчитать ожидаемые выплаты настоящим калькулятором, а не выдумать их.
Это даёт нам два эффекта:
  1. Числа payoff в трейсе — реальные (детерминированные), не галлюцинированные.
  2. Мы можем поймать расхождение между тем, что модель «насчитала словами»,
    и тем, что вернул инструмент (проверка галлюцинаций по числам).

Каждый результат инструмента несёт поле source="tool" — честная пометка
происхождения числа, как в семинаре 5.
"""
from __future__ import annotations

from statistics import mean

PIE = 100


def expected_payoffs(offer: int, accept_prob: float) -> dict:
    """
    Ожидаемые выплаты обоих игроков при данном предложении и вероятности приёма.

    Args:
        offer: сколько отдаём Отвечающему (0..100).
        accept_prob: вероятность, что предложение примут (0..1).

    Returns:
        {"offer": int, "proposer_keep": int, "ev_proposer": float,
         "ev_responder": float, "source": "tool"}
    """
    if not 0 <= offer <= PIE:
        raise ValueError(f"offer={offer} вне 0..{PIE}")
    if not 0.0 <= accept_prob <= 1.0:
        raise ValueError(f"accept_prob={accept_prob} вне 0..1")
    keep = PIE - offer
    return {
        "offer": offer,
        "proposer_keep": keep,
        "ev_proposer": round(keep * accept_prob, 2),
        "ev_responder": round(offer * accept_prob, 2),
        "source": "tool",
    }


def settle_round(offer: int, accept: bool) -> dict:
    """
    Детерминированный расчёт фактических выплат после хода Отвечающего.

    Returns:
        {"proposer_payoff": int, "responder_payoff": int, "source": "tool"}
    """
    if not 0 <= offer <= PIE:
        raise ValueError(f"offer={offer} вне 0..{PIE}")
    if accept:
        return {
            "proposer_payoff": PIE - offer,
            "responder_payoff": offer,
            "source": "tool",
        }
    return {"proposer_payoff": 0, "responder_payoff": 0, "source": "tool"}


def acceptance_stats(history: list[dict]) -> dict:
    """
    Сводка по истории: средняя принятая доля и грубая оценка вероятности приёма.

    history — список dict с ключами offer:int, accept:bool.
    Возвращает source="tool".
    """
    if not history:
        return {"n": 0, "accept_rate": 0.0, "mean_accepted_offer": 0.0, "source": "tool"}
    accepts = [h for h in history if h["accept"]]
    return {
        "n": len(history),
        "accept_rate": round(len(accepts) / len(history), 3),
        "mean_accepted_offer": round(mean([h["offer"] for h in accepts]), 1)
        if accepts
        else 0.0,
        "source": "tool",
    }


# Реестр инструментов — агент выбирает по имени (как в ReAct семинара 5).
TOOLS = {
    "expected_payoffs": expected_payoffs,
    "settle_round": settle_round,
    "acceptance_stats": acceptance_stats,
}
