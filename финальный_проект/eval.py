"""
Eval проекта «Ультиматум: homo silicus».

Прогоняет тестовые входы (input/test_cases.json) через пайплайн и считает:

ПРАВИЛЬНОСТЬ:
  * exact_match  — совпало ли решение модели (accept) с ground-truth порогом персоны.
  * judge_consistent_rate — доля ходов, признанных судьёй согласованными.
  * judge_mean_score — средняя оценка рациональности 1..5.

ГИПОТЕЗА (трек A):
  * reject_rate_model(offer) vs reject_rate_human(offer) по 5 уровням предложения.
  * 95% доверительные интервалы Вилсона по каждому уровню.

ПУТЬ:
  * avg_steps, tools_used, total_tokens, est_cost_usd.

ДИАГНОСТИКА:
  * hallucination_report.json — подробный отчёт по галлюцинациям.
  * failures.json — провалы exact-match.

Запуск:
    python eval.py            # полный прогон через API
    python eval.py --offline  # офлайн-симуляция (проверка кода без API)
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"
OUTPUT = HERE / "output"
OUTPUT.mkdir(exist_ok=True)

PRICE_IN = 0.07 / 1_000_000
PRICE_OUT = 0.28 / 1_000_000


def load_cases() -> list[dict]:
    p = INPUT / "test_cases.json"
    if not p.exists():
        raise SystemExit("Сначала: python make_input.py")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Доверительный интервал Вилсона (95%) для доли
# ---------------------------------------------------------------------------

def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% ДИ Вилсона для k успехов из n испытаний."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (round(max(0.0, center - spread), 3), round(min(1.0, center + spread), 3))


# ---------------------------------------------------------------------------
# Офлайн-симуляция
# ---------------------------------------------------------------------------

def _offline_round(persona, offer, round_num, history):
    from personas import get_persona
    from pipeline import RoundTrace
    from tools import settle_round

    p = get_persona(persona)
    accept = offer >= p.reject_below
    fairness = "fair" if offer >= 40 else ("borderline" if offer >= 25 else "unfair")
    settled = settle_round(offer, accept)
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


# ---------------------------------------------------------------------------
# Основной eval
# ---------------------------------------------------------------------------

def run_eval(offline: bool) -> dict:
    cases = load_cases()
    rows = []
    failures = []
    halluc_cases = []   # для hallucination_report
    halluc_total = 0
    tool_counter: Counter = Counter()
    step_counts = []
    tok_in = tok_out = 0

    # для гипотезы: reject по уровню предложения
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
                "repeat": c.get("repeat", 1),
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
        # галлюцинации
        if tr.hallucinations:
            halluc_cases.append({
                "id": c["id"],
                "persona_id": pid,
                "offer": offer,
                "flags": tr.hallucinations,
            })
        halluc_total += len(tr.hallucinations)
        for t in tr.tools_called:
            tool_counter[t] += 1
        step_counts.append(len(tr.steps))
        tok_in += tr.prompt_tokens
        tok_out += tr.completion_tokens
        reject_by_offer.setdefault(offer, []).append(not tr.accept)

        print(
            f"[{i:3d}/{len(cases)}] {c['id']} {pid:18s} off={offer:2d} "
            f"-> accept={tr.accept} exact={exact} judge={tr.judge_score}"
        )

    n = len(rows)
    exact_rate = sum(r["exact_match"] for r in rows) / n
    consistent_rate = sum(r["judge_consistent"] for r in rows) / n
    mean_judge = sum(r["judge_score"] for r in rows) / n

    # --- Гипотеза: reject_rate модели vs людей + ДИ Вилсона ---
    bench = json.loads((INPUT / "human_benchmark.json").read_text(encoding="utf-8"))
    human_rr = bench["reject_rate_by_offer"]

    hypothesis_rows = []
    for off in sorted(reject_by_offer.keys()):
        vals = reject_by_offer[off]
        k = sum(vals)
        nn = len(vals)
        model_rr = round(k / nn, 3)
        ci = wilson_ci(k, nn)
        human = human_rr.get(str(off), [None, None])
        in_human_range = (
            human[0] is not None and human[0] <= model_rr <= human[1]
        )
        hypothesis_rows.append({
            "offer": off,
            "n": nn,
            "model_reject_rate": model_rr,
            "ci_95": list(ci),
            "human_range": human,
            "in_human_range": in_human_range,
        })

    verdict = _hypothesis_verdict(hypothesis_rows)

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
            "by_offer": hypothesis_rows,
            "human_benchmark_source": bench["source"],
            "verdict": verdict,
        },
    }

    (OUTPUT / "eval_results.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT / "failures.json").write_text(
        json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # --- hallucination_report.json ---
    halluc_report = {
        "total_caught": halluc_total,
        "n_cases_with_hallucinations": len(halluc_cases),
        "rate_per_case": round(halluc_total / n, 3),
        "cases": halluc_cases,
        "note": (
            "ghost_round — ссылка на несуществующий раунд; "
            "ghost_number — число в reasoning не совпадает с offer/keep/0/50/100."
        ),
    }
    (OUTPUT / "hallucination_report.json").write_text(
        json.dumps(halluc_report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_table(rows, summary)
    return summary


def _hypothesis_verdict(rows: list[dict]) -> str:
    """
    Справедливый вердикт: воспроизводит ли модель направление эффекта?
    Избегаем категоричного 'ПОДТВЕРЖДЕНО' — абсолютные пороги требуют проверки.
    """
    low = next((r["model_reject_rate"] for r in rows if r["offer"] == 10), None)
    high = next((r["model_reject_rate"] for r in rows if r["offer"] == 50), None)
    in_range_count = sum(1 for r in rows if r["in_human_range"])
    total = len(rows)

    if low is not None and high is not None and low > high:
        return (
            f"Модель воспроизводит направление эффекта: reject_rate при offer=10 ({low:.0%}) "
            f"> reject_rate при offer=50 ({high:.0%}), как у людей. "
            f"Однако абсолютные пороги требуют проверки: "
            f"только {in_range_count}/{total} уровней предложения попали в человеческий диапазон. "
            f"Вывод: модель качественно согласуется с поведением людей, "
            f"но количественные расхождения указывают на RLHF-смещение."
        )
    return (
        "Модель не воспроизводит человеческий паттерн отвержения: "
        "reject_rate не убывает монотонно с ростом предложения."
    )


def _write_table(rows: list[dict], summary: dict) -> None:
    lines = [
        "| id    | persona            | offer | rep | exp | got | exact | judge | halluc |",
        "|-------|--------------------|-------|-----|-----|-----|-------|-------|--------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['persona_id']:18s} | {r['offer']:5d} | "
            f"{r.get('repeat',1):3d} | "
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
    print(f"\nАртефакты: output/eval_results.json, eval_table.md, failures.json, hallucination_report.json")


if __name__ == "__main__":
    main()
