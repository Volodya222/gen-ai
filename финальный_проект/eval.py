"""
Eval проекта «Ультиматум: homo silicus».

Прогоняет 15 тестовых входов (input/test_cases.json) через пайплайн и считает:

ПРАВИЛЬНОСТЬ:
  * exact_match  — совпало ли решение модели (accept) с ground-truth порогом
                   персоны (expected_accept).
  * judge_consistent_rate — доля ходов, которые судья признал согласованными.
  * judge_mean_score — средняя оценка рациональности 1..5.

ПУТЬ (требование рубрики, блок 3):
  * avg_steps        — среднее число шагов пайплайна на вход.
  * tools_used       — какие инструменты звались и сколько раз.
  * total_tokens     — суммарно prompt+completion (proxy стоимости).
  * est_cost_usd     — грубая оценка стоимости по цене DeepSeek-flash.

ДИАГНОСТИКА:
  * hallucinations_caught — сколько ghost-раундов/чисел поймано.
  * провалы (вход → искажение) выписываются в output/failures.json.

ГИПОТЕЗА (трек A): воспроизводит ли модель человеческий стилизованный факт —
  отвержение низких предложений? Сравниваем с input/human_benchmark.json.

Запуск:
    python eval.py            # полный прогон (15 входов)
    python eval.py --offline  # офлайн-симуляция без API (для самопроверки кода)
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

# Цена DeepSeek-flash (вход/выход за 1M токенов) — для оценки стоимости.
PRICE_IN = 0.07 / 1_000_000
PRICE_OUT = 0.28 / 1_000_000


def load_cases() -> list[dict]:
    p = INPUT / "test_cases.json"
    if not p.exists():
        raise SystemExit("Сначала: python make_input.py")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Офлайн-симуляция (детерминированная) — чтобы прогнать структуру eval без API.
# С реальным токеном используется боевой путь (run_round из pipeline).
# ---------------------------------------------------------------------------


def _offline_round(persona, offer, round_num, history):
    from personas import get_persona
    from pipeline import RoundTrace
    from tools import settle_round

    p = get_persona(persona)
    accept = offer >= p.reject_below
    fairness = "fair" if offer >= 40 else ("borderline" if offer >= 25 else "unfair")
    settled = settle_round(offer, accept)
    # детерминированный «судья»: согласованно, если решение бьётся с восприятием
    consistent = not (fairness == "unfair" and accept and offer < 10)
    return RoundTrace(
        round=round_num,
        persona_id=persona,
        offer=offer,
        steps=["persona_conditioning", "responder_decide", "settle", "judge", "hallucination_check"],
        tools_called=["settle_round"],
        accept=accept,
        fairness_perceived=fairness,
        proposer_payoff=settled["proposer_payoff"],
        responder_payoff=settled["responder_payoff"],
        judge_score=5 if consistent else 2,
        judge_consistent=consistent,
        hallucinations=[],
        prompt_tokens=180,
        completion_tokens=40,
    )


def run_eval(offline: bool) -> dict:
    cases = load_cases()
    rows = []
    failures = []
    halluc_total = 0
    tool_counter: Counter = Counter()
    step_counts = []
    tok_in = tok_out = 0

    # для проверки гипотезы: отвержения по уровню предложения
    reject_by_offer: dict[int, list[bool]] = {}

    if not offline:
        from personas import get_persona
        from pipeline import run_round

    for i, c in enumerate(cases, 1):
        pid, offer = c["persona_id"], c["offer"]
        if offline:
            tr = _offline_round(pid, offer, 1, [])
        else:
            tr = run_round(get_persona(pid), offer, 1, [])

        exact = tr.accept == c["expected_accept"]
        rows.append(
            {
                "id": c["id"],
                "persona_id": pid,
                "offer": offer,
                "expected_accept": c["expected_accept"],
                "model_accept": tr.accept,
                "exact_match": exact,
                "fairness_perceived": tr.fairness_perceived,
                "judge_score": tr.judge_score,
                "judge_consistent": tr.judge_consistent,
                "n_steps": len(tr.steps),
                "tools": tr.tools_called,
                "hallucinations": tr.hallucinations,
                "tokens": tr.prompt_tokens + tr.completion_tokens,
            }
        )
        if not exact:
            failures.append(
                {
                    "id": c["id"],
                    "input": {"persona": pid, "offer": offer, "segment": c["segment"]},
                    "expected_accept": c["expected_accept"],
                    "got_accept": tr.accept,
                    "fairness_perceived": tr.fairness_perceived,
                    "distortion": (
                        "модель отвергла там, где сегмент обычно принимает"
                        if c["expected_accept"]
                        else "модель приняла там, где сегмент обычно отвергает"
                    ),
                }
            )
        halluc_total += len(tr.hallucinations)
        for t in tr.tools_called:
            tool_counter[t] += 1
        step_counts.append(len(tr.steps))
        tok_in += tr.prompt_tokens
        tok_out += tr.completion_tokens
        reject_by_offer.setdefault(offer, []).append(not tr.accept)

        print(
            f"[{i:2d}/{len(cases)}] {c['id']} {pid:18s} off={offer:2d} "
            f"-> accept={tr.accept} exact={exact} judge={tr.judge_score}"
        )

    n = len(rows)
    exact_rate = sum(r["exact_match"] for r in rows) / n
    consistent_rate = sum(r["judge_consistent"] for r in rows) / n
    mean_judge = sum(r["judge_score"] for r in rows) / n

    # проверка гипотезы: доля отвержений по уровню предложения
    reject_rates = {
        off: round(sum(v) / len(v), 3) for off, v in sorted(reject_by_offer.items())
    }

    bench = json.loads((INPUT / "human_benchmark.json").read_text(encoding="utf-8"))

    summary = {
        "n_cases": n,
        "correctness": {
            "exact_match_rate": round(exact_rate, 3),
            "judge_consistent_rate": round(consistent_rate, 3),
            "judge_mean_score": round(mean_judge, 2),
            "pass_rate": round(exact_rate, 3),
        },
        "path": {
            "avg_steps": round(sum(step_counts) / n, 2),
            "tools_used": dict(tool_counter),
            "total_tokens": tok_in + tok_out,
            "prompt_tokens": tok_in,
            "completion_tokens": tok_out,
            "est_cost_usd": round(tok_in * PRICE_IN + tok_out * PRICE_OUT, 6),
        },
        "hallucinations": {
            "total_caught": halluc_total,
            "rate_per_case": round(halluc_total / n, 3),
        },
        "hypothesis_check": {
            "reject_rate_by_offer_pct": reject_rates,
            "human_benchmark": bench["low_offer_reject_rate"],
            "verdict": _hypothesis_verdict(reject_rates),
        },
    }

    (OUTPUT / "eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_table(rows, summary)
    return summary


def _hypothesis_verdict(reject_rates: dict[int, float]) -> str:
    """Воспроизводится ли стилизованный факт: низкие предложения отвергаются чаще."""
    low = reject_rates.get(10, 0.0)
    high = reject_rates.get(45, 0.0)
    if low > high and low >= 0.3:
        return (
            "ПОДТВЕРЖДЕНО: модель отвергает низкие предложения заметно чаще, "
            "чем почти-равные — как люди в экспериментах."
        )
    if low > high:
        return "ЧАСТИЧНО: тренд верный, но отвержение слабее человеческого."
    return "НЕ ПОДТВЕРЖДЕНО: модель не воспроизводит человеческий паттерн отвержения."


def _write_table(rows: list[dict], summary: dict) -> None:
    lines = [
        "| id   | persona            | offer | exp | got | exact | judge | halluc |",
        "|------|--------------------|-------|-----|-----|-------|-------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['persona_id']:18s} | {r['offer']:5d} | "
            f"{str(r['expected_accept'])[0]}   | {str(r['model_accept'])[0]}   | "
            f"{'✓' if r['exact_match'] else '✗':5s} | {r['judge_score']:5d} | "
            f"{len(r['hallucinations']):6d} |"
        )
    lines.append("")
    c = summary["correctness"]
    p = summary["path"]
    lines.append(
        f"pass_rate={c['pass_rate']} | judge_consistent={c['judge_consistent_rate']} | "
        f"judge_mean={c['judge_mean_score']} | avg_steps={p['avg_steps']} | "
        f"tokens={p['total_tokens']} | cost=${p['est_cost_usd']} | "
        f"halluc={summary['hallucinations']['total_caught']}"
    )
    (OUTPUT / "eval_table.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    offline = "--offline" in sys.argv
    print(f"=== EVAL ({'offline-симуляция' if offline else 'боевой прогон через API'}) ===")
    summary = run_eval(offline)
    print("\n=== ИТОГ ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nАртефакты: output/eval_results.json, eval_table.md, failures.json")


if __name__ == "__main__":
    main()
