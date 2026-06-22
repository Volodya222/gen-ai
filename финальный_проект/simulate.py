"""
Свободная мульти-агентная симуляция: Предлагающий-агент (с инструментами)
против синтетической персоны-Отвечающего, N раундов с памятью истории.

В отличие от eval.py (фиксированная сетка предложений), здесь Предлагающий сам
выбирает предложение, опираясь на калькулятор выплат и историю принятий —
полноценный агент с инструментами в мультиагентном цикле.

Запуск:
    python simulate.py                      # persona=student_fair, 6 раундов
    python simulate.py poor_pragmatist 8    # другая персона, 8 раундов
    python simulate.py student_fair 6 --offline
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent / "output"
OUTPUT.mkdir(exist_ok=True)


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    offline = "--offline" in sys.argv
    persona_id = args[0] if args else "student_fair"
    rounds = int(args[1]) if len(args) > 1 else 6

    from personas import get_persona

    persona = get_persona(persona_id)
    history: list[dict] = []
    transcript = []

    print(f"=== СИМУЛЯЦИЯ: Предлагающий vs {persona_id} ({rounds} раундов)"
          f"{' [offline]' if offline else ''} ===")

    for rnd in range(1, rounds + 1):
        if offline:
            # детерминированный предлагающий: учится повышать предложение при отказах
            last_rejects = sum(1 for h in history if not h["accept"])
            offer = min(50, 20 + 5 * last_rejects)
            tools = ["acceptance_stats", "expected_payoffs"]
            from eval import _offline_round
            tr = _offline_round(persona_id, offer, rnd, history)
        else:
            from pipeline import proposer_decide, run_round
            move, tools, _ = proposer_decide(rnd, history)
            offer = move.share_to_responder
            tr = run_round(persona, offer, rnd, history)

        history.append({"round": rnd, "offer": offer, "accept": tr.accept})
        transcript.append(
            {
                "round": rnd,
                "proposer_tools": tools,
                "offer": offer,
                "responder_accept": tr.accept,
                "fairness_perceived": tr.fairness_perceived,
                "proposer_payoff": tr.proposer_payoff,
                "responder_payoff": tr.responder_payoff,
                "judge_score": tr.judge_score,
                "hallucinations": tr.hallucinations,
            }
        )
        print(
            f"  раунд {rnd}: offer={offer:2d} -> "
            f"{'принято' if tr.accept else 'ОТКАЗ'} "
            f"(P={tr.proposer_payoff}, R={tr.responder_payoff})"
        )

    out = OUTPUT / f"transcript_{persona_id}.json"
    out.write_text(
        json.dumps(
            {"persona": persona_id, "rounds": rounds, "transcript": transcript},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    accepts = sum(1 for h in history if h["accept"])
    print(f"\nПринято {accepts}/{rounds}. Транскрипт: {out}")


if __name__ == "__main__":
    main()
