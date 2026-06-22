"""
Макро-агент: ReAct + вызов инструментов:
  блоки 1-3 — базовый агент: исполняем инструменты, отвечает текстом.
  блок 5    — новый инструмент get_unemployment + тяжёлые multi-hop задачи.
  блок 6    — параллельные вызовы инструментов (флаг --parallel).
  блок 7    — структурированный ответ через submit_answer (флаг --structured).
  блок 8    — самопроверка перед ответом (флаг --critic).
  блок 9    — кэш детерминированных инструментов (флаг --cache).
  блок 10   — учёт токенов и стоимости (флаг --cost).

Запуск:
    python agent.py "Какая реальная ключевая ставка сейчас?"
    python agent.py --parallel --structured --critic "Сравни курс USD сегодня и 2 января 2022"
    python agent.py --cost "Что сейчас выше: ключевая ставка или индекс нищеты?"

"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

import uuid

from llm_client import get_model, make_client, make_raw_client
from schemas import TOOL_SCHEMAS
from tools import (
    calculate,
    compare_periods,
    get_fx_rate,
    get_inflation,
    get_key_rate,
    get_unemployment,
)

# набор инструментов
TOOLS_IMPL = {
    "get_fx_rate": get_fx_rate,
    "get_key_rate": get_key_rate,
    "get_inflation": get_inflation,
    "get_unemployment": get_unemployment,
    "calculate": calculate,
    "compare_periods": compare_periods,  # ← ДЗ семинара 5
}

# ── JSONL-лог трасс (ДЗ семинара 5): все шаги каждого прогона ──
TRACE_PATH = Path(__file__).resolve().parent / "trace.jsonl"


def _now_iso() -> str:
    return datetime.datetime.now().replace(microsecond=0).isoformat()


def append_jsonl(record: dict, path: Path = TRACE_PATH) -> None:
    """Дописать одну строку события в trace.jsonl (режим 'a' — логи копятся)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# блок 7 — структурированный ответ
class AgentAnswer(BaseModel):
    answer: str = Field(description="Ответ человеку, одна-две фразы")
    value: Optional[float] = Field(default=None, description="Главное число ответа")
    unit: Optional[str] = Field(default=None, description="Единица: %, руб, год")
    sources: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


SUBMIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "submit_answer",
        "description": "Вызови ТОЛЬКО когда данных достаточно для финального ответа. "
        "Передай ответ структурой, не текстом.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "value": {"type": ["number", "null"]},
                "unit": {"type": ["string", "null"]},
                "sources": {"type": "array", "items": {"type": "string"}},
                "confidence": {"type": "number"},
            },
            "required": ["answer", "confidence"],
        },
    },
}


# блок 8 — самопроверка
class CriticVerdict(BaseModel):
    ok: bool
    issue: str = ""


CRITIC_SYSTEM = """Ты — придирчивый ревизор. Тебе дают финальный ответ агента и
лог инструментов. Проверь ОДНО: выводится ли число в ответе из данных
инструментов, без выдумки. ok=false, если число не подтверждается логом или
арифметика не сходится. issue — одна фраза, что не так."""


# блок 9 — кэш детерминированных инструментов (живёт в пределах процесса).
TOOL_CACHE: dict[str, dict] = {}
CACHE_STATS = {"hits": 0, "misses": 0}

# блок 10 — грубая оценка стоимости. Цена за 1 млн токенов, USD (ориентир DeepSeek).
PRICE_IN_PER_MTOK = 0.14
PRICE_OUT_PER_MTOK = 0.28


_BASE_RULES = """\
Ты — макроэкономический аналитик с данными Цб РФ и Росстата. ЧИСЛА НИКОГДА НЕ
ПРИДУМЫВАЙ — получай их через инструменты.

Инструменты:
- get_fx_rate: курс валюты к рублю на дату
- get_key_rate: ключевая ставка Цб на дату
- get_inflation: ИПЦ (% г/г) на конец месяца
- get_unemployment: безработица (% рабочей силы) на конец месяца
- calculate: безопасный калькулятор для арифметики над полученными числами

Алгоритм:
1. Разложи вопрос: какие числа нужны и в каком порядке. Если несколько чисел
   независимы — запрашивай их в одном шаге (несколько вызовов сразу).
2. Арифметику считай ТОЛЬКО через calculate.
3. Реальная ставка = номинальная ставка − инфляция г/г.
4. Реальная доходность вклада ≈ (1 + ставка/100) / (1 + инфляция/100) − 1.
5. Индекс нищеты = инфляция г/г + безработица.
6. Кросс-курс «сколько B за 1 A» = (рублей за 1 A) / (рублей за 1 B).
   Пример: «юаней за доллар» = (рублей за доллар) / (рублей за юань).
"""

SYSTEM_PROMPT = (
    _BASE_RULES
    + """\
7. Когда данных достаточно — выдай финальный ответ обычным текстом бЕЗ вызовов
   инструментов. Одна-две фразы, с числами и единицами. Если число из
   fallback_csv — оговорись, что Цб в моменте недоступен.
Формат даты — YYYY-MM-DD.
Текущая дата: {}
""".format(datetime.datetime.now().strftime("%Y-%m-%d"))
)

SYSTEM_PROMPT_PRO = (
    _BASE_RULES
    + """\
7. Когда данных достаточно — НЕ пиши текст, а вызови submit_answer со структурой
   (answer, value, unit, sources, confidence).
Формат даты — YYYY-MM-DD.
"""
)


def _exec_one(tc, cache: Optional[dict] = None) -> tuple[Any, dict, dict]:
    """Исполнить один вызов инструмента. Вернуть (tc, args, obs).
    Любую ошибку превращаем в obs={'error': ...}, чтобы агент мог переиграть."""
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments or "{}")
    except JSONDecodeError as e:
        return tc, {}, {"error": f"битый json аргументов: {e}"}

    fn = TOOLS_IMPL.get(name)
    if fn is None:
        return tc, args, {"error": f"неизвестный инструмент: {name}"}

    key = name + ":" + json.dumps(args, sort_keys=True, ensure_ascii=False)
    if cache is not None and key in cache:
        CACHE_STATS["hits"] += 1
        return tc, args, cache[key]

    try:
        obs = fn(**args)
    except TypeError as e:
        return (
            tc,
            args,
            {
                "error": f"плохие аргументы для {name}: {e}. Expected: {fn.__annotations__}"
            },
        )
    except Exception as e:
        return tc, args, {"error": f"{type(e).__name__}: {e}"}

    if cache is not None and "error" not in obs:
        CACHE_STATS["misses"] += 1
        cache[key] = obs
    return tc, args, obs


def critique(answer: AgentAnswer, tool_log: list[dict]) -> CriticVerdict:
    ic = make_client()
    facts = "\n".join(
        f"{e['call']}({e['args']}) -> {json.dumps(e['obs'], ensure_ascii=False)}"
        for e in tool_log
        if "call" in e
    )
    return ic.chat.completions.create(
        model=get_model(),
        response_model=CriticVerdict,
        max_retries=2,
        temperature=0.0,
        messages=[
            {"role": "system", "content": CRITIC_SYSTEM},
            {
                "role": "user",
                "content": f"Ответ агента: «{answer.answer}» (value={answer.value} {answer.unit}).\n"
                f"Лог инструментов:\n{facts or '(пусто)'}",
            },
        ],
    )


def _finish(
    res: dict,
    usage_log: list[dict],
    *,
    track_cost: bool,
    use_cache: bool,
    verbose: bool,
) -> dict:
    """Прикрепить к результату учёт токенов/стоимости (блок 10) и статистику
    кэша (блок 9); по флагам — распечатать. Этот код готов; чтобы таблица
    стоимости заполнилась, надо заполнить usage_log в run_agent (блок 10)."""
    total_in = sum(u["prompt_tokens"] for u in usage_log)
    total_out = sum(u["completion_tokens"] for u in usage_log)
    total_cost = round(sum(u["cost_usd"] for u in usage_log), 6)
    res["usage"] = {
        "prompt_tokens": total_in,
        "completion_tokens": total_out,
        "cost_usd": total_cost,
        "by_step": usage_log,
    }
    if use_cache:
        res["cache"] = dict(CACHE_STATS)

    if track_cost and usage_log:
        print("\n  шаг | вход.ток | выход.ток |   $/шаг |  $ накоп.")
        acc = 0.0
        for u in usage_log:
            acc += u["cost_usd"]
            print(
                f"  {u['step']:>3} | {u['prompt_tokens']:>8} | {u['completion_tokens']:>9} | "
                f"{u['cost_usd']:.5f} | {acc:.5f}"
            )
        print(
            f"  Итого: {total_in} вход + {total_out} выход токенов, ~${total_cost:.5f}."
        )
    if use_cache and verbose:
        print(
            f"  [кэш] попаданий {CACHE_STATS['hits']}, промахов {CACHE_STATS['misses']}"
        )
    return res


def run_agent(
    user_query: str,
    *,
    max_iter: int = 8,
    parallel: bool = False,
    structured: bool = False,
    use_critic: bool = False,
    use_cache: bool = False,
    track_cost: bool = False,
    verbose: bool = True,
) -> dict[str, Any]:
    """ReAct-цикл. базовый режим — финал текстом; флаги включают блоки 6-10."""
    run_id = uuid.uuid4().hex[:12]
    try:
        client = make_raw_client()
        model = get_model()
    except Exception as e:
        if verbose:
            print(f"[offline] LLM недоступен ({type(e).__name__}: {e}). "
                  f"Включаю детерминированный планировщик (инструменты работают через CSV).")
        return _offline_run(user_query, run_id=run_id, structured=structured, verbose=verbose)
    tools = TOOL_SCHEMAS + ([SUBMIT_SCHEMA] if structured else [])
    system = SYSTEM_PROMPT_PRO if structured else SYSTEM_PROMPT
    cache = TOOL_CACHE if use_cache else None
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_query},
    ]
    trace: list[dict[str, Any]] = []
    usage_log: list[dict[str, Any]] = []  # блок 10 — токены по шагам

    for step in range(1, max_iter + 1):
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.0,
        )
        msg = resp.choices[0].message
        messages.append(msg.model_dump(exclude_none=True))

        # блок 10 — учёт токенов шага
        u = getattr(resp, "usage", None)
        if u is not None:
            pin, pout = u.prompt_tokens, u.completion_tokens
            cost = pin / 1e6 * PRICE_IN_PER_MTOK + pout / 1e6 * PRICE_OUT_PER_MTOK
            usage_log.append(
                {
                    "step": step,
                    "prompt_tokens": pin,
                    "completion_tokens": pout,
                    "cost_usd": round(cost, 6),
                }
            )

        if verbose:
            names = [tc.function.name for tc in (msg.tool_calls or [])]
            print(f"[step {step}] {names or 'финал-текст'}")

        if not msg.tool_calls:
            trace.append({"step": step, "final": msg.content})
            append_jsonl({"run_id": run_id, "ts": _now_iso(), "step": step,
                          "final": msg.content})
            return _finish(
                {
                    "answer": msg.content,
                    "structured": None,
                    "trace": trace,
                    "steps": step,
                },
                usage_log,
                track_cost=track_cost,
                use_cache=use_cache,
                verbose=verbose,
            )

        submit = next(
            (tc for tc in msg.tool_calls if tc.function.name == "submit_answer"), None
        )
        others = [tc for tc in msg.tool_calls if tc is not submit]

        # блок 6 — исполняем обычные вызовы (параллельно, если их несколько)
        if others:
            if parallel and len(others) > 1:
                with ThreadPoolExecutor(max_workers=4) as ex:
                    results = list(ex.map(lambda t: _exec_one(t, cache), others))
            else:
                results = [_exec_one(tc, cache) for tc in others]
            for tc, args, obs in results:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(obs, ensure_ascii=False),
                    }
                )
                trace.append(
                    {"step": step, "call": tc.function.name, "args": args, "obs": obs}
                )
                append_jsonl({
                    "run_id": run_id, "ts": _now_iso(), "step": step,
                    "call": tc.function.name, "args": args, "obs": obs,
                })
                if verbose:
                    print(
                        f"    {tc.function.name}({args}) -> {json.dumps(obs, ensure_ascii=False)[:140]}"
                    )

        # блок 7 + 8 — финал через submit_answer и самопроверку
        if submit is not None:
            try:
                ans = AgentAnswer(**json.loads(submit.function.arguments or "{}"))
            except Exception as e:
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": submit.id,
                        "content": f"submit_answer невалиден: {e}. Исправь.",
                    }
                )
                continue
            if use_critic:
                verdict = critique(ans, trace)
                if verbose:
                    print(f"    [ревизор] ok={verdict.ok} {verdict.issue}")
                if not verdict.ok:
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": submit.id,
                            "content": f"Ревизор отклонил: {verdict.issue}. "
                            f"Перепроверь и вызови submit_answer заново.",
                        }
                    )
                    continue
            messages.append(
                {"role": "tool", "tool_call_id": submit.id, "content": "ответ принят"}
            )
            append_jsonl({"run_id": run_id, "ts": _now_iso(), "step": step,
                          "final": ans.answer})
            return _finish(
                {
                    "answer": ans.answer,
                    "structured": ans,
                    "trace": trace,
                    "steps": step,
                },
                usage_log,
                track_cost=track_cost,
                use_cache=use_cache,
                verbose=verbose,
            )

    return _finish(
        {
            "answer": None,
            "structured": None,
            "trace": trace,
            "steps": max_iter,
            "error": f"исчерпан лимит шагов max_iter={max_iter}",
        },
        usage_log,
        track_cost=track_cost,
        use_cache=use_cache,
        verbose=verbose,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query", nargs="+", help="Вопрос к агенту")
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="блок 6: параллельные вызовы инструментов",
    )
    ap.add_argument(
        "--structured",
        action="store_true",
        help="блок 7: структурный финал через submit_answer",
    )
    ap.add_argument(
        "--critic",
        action="store_true",
        help="блок 8: самопроверка финала (нужен --structured)",
    )
    ap.add_argument(
        "--cache",
        action="store_true",
        help="блок 9: кэш детерминированных инструментов",
    )
    ap.add_argument(
        "--cost",
        action="store_true",
        help="блок 10: показать токены и стоимость по шагам",
    )
    ap.add_argument("--trace", type=Path, default=None, help="Куда сохранить JSON-лог")
    a = ap.parse_args()

    q = " ".join(a.query)
    res = run_agent(
        q,
        max_iter=a.max_iter,
        verbose=not a.quiet,
        parallel=a.parallel,
        structured=a.structured,
        use_critic=a.critic,
        use_cache=a.cache,
        track_cost=a.cost,
    )

    print("\n=== ВОПРОС ===")
    print(q)
    print("\n=== ОТВЕТ ===")
    s = res.get("structured")
    if s:
        print(s.answer)
        print(
            f"value={s.value} {s.unit or ''} | sources={s.sources} | confidence={s.confidence:.2f}"
        )
    else:
        print(res.get("answer") or res.get("error"))
    print(f"\n(шагов: {res['steps']})")


# ===========================================================================
# Офлайн-планировщик (ДЗ): когда LLM-ключа нет, детерминированно разбираем
# вопрос, дёргаем РЕАЛЬНЫЕ инструменты (они отвечают из CSV-fallback) и пишем
# тот же trace.jsonl. С рабочим ключом этот код не вызывается — работает ReAct.
# ===========================================================================
_MONTHS = [
    ("январ", 1), ("феврал", 2), ("март", 3), ("апрел", 4), ("ма[йя]", 5),
    ("июн", 6), ("июл", 7), ("август", 8), ("сентябр", 9), ("октябр", 10),
    ("ноябр", 11), ("декабр", 12),
]
_MONTH_RE = "|".join(f"{stem}\\w*" for stem, _ in _MONTHS)


def _month_num(word: str) -> int:
    for stem, num in _MONTHS:
        if re.match(stem, word.lower()):
            return num
    return 0


def _extract_periods(q: str) -> list[str]:
    """Вернуть периоды в порядке появления: 'YYYY-MM-DD' / 'YYYY-MM'."""
    out: list[tuple[int, str]] = []
    for m in re.finditer(r"(\d{4})-(\d{2})(?:-(\d{2}))?", q):
        out.append((m.start(), m.group(0)))
    for m in re.finditer(rf"(\d{{1,2}})\s+({_MONTH_RE})\s+(\d{{4}})", q):
        day, mon, year = int(m.group(1)), _month_num(m.group(2)), m.group(3)
        out.append((m.start(), f"{year}-{mon:02d}-{day:02d}"))
    for m in re.finditer(rf"({_MONTH_RE})\s+(\d{{4}})", q):
        # пропустим, если этот месяц уже учтён как часть «день месяц год»
        if any(abs(pos - m.start()) < 4 for pos, _ in out):
            continue
        mon, year = _month_num(m.group(1)), m.group(2)
        out.append((m.start(), f"{year}-{mon:02d}"))
    seen, res = set(), []
    for _, p in sorted(out):
        if p not in seen:
            seen.add(p)
            res.append(p)
    return res


def _currency(q: str) -> str | None:
    ql = q.lower()
    if "usd" in ql or "доллар" in ql:
        return "USD"
    if "eur" in ql or "евро" in ql:
        return "EUR"
    if "cny" in ql or "юан" in ql:
        return "CNY"
    return None


def _latest(fn, key: str) -> tuple[int, int, float] | None:
    y, m = 2026, 6
    for _ in range(20):
        r = fn(y, m)
        if "error" not in r:
            return y, m, r[key]
        m -= 1
        if m == 0:
            y, m = y - 1, 12
    return None


def _offline_run(user_query: str, *, run_id: str, structured: bool, verbose: bool) -> dict:
    q = user_query
    ql = q.lower()
    trace: list[dict] = []
    step = 0

    def emit(call, args, obs):
        nonlocal step
        step += 1
        trace.append({"step": step, "call": call, "args": args, "obs": obs})
        append_jsonl({"run_id": run_id, "ts": _now_iso(), "step": step,
                      "call": call, "args": args, "obs": obs})
        if verbose:
            print(f"  [{step}] {call}({args}) -> {json.dumps(obs, ensure_ascii=False)[:120]}")
        return obs

    def finish(text):
        trace.append({"step": step + 1, "final": text})
        append_jsonl({"run_id": run_id, "ts": _now_iso(), "step": step + 1, "final": text})
        if verbose:
            print(f"  [final] {text}")
        return {"answer": text, "structured": None, "trace": trace, "steps": step + 1}

    def note(obs) -> str:
        src = obs.get("source", "")
        return " (ЦБ в моменте недоступен, значение из локального архива)" if "fallback" in src else ""

    periods = _extract_periods(q)
    cur = _currency(q)

    # 1) Сравнение двух периодов
    if re.search(r"во сколько раз|на сколько (вырос|измен|подорожал|снизил|упал)|сравни|по сравнению", ql) and len(periods) >= 2:
        if cur:
            metric = f"fx_{cur}"
        elif "ставк" in ql:
            metric = "key_rate"
        elif "инфляц" in ql or "ипц" in ql:
            metric = "cpi"
        elif "безработиц" in ql:
            metric = "unemployment"
        else:
            metric = "fx_USD"
        obs = emit("compare_periods",
                   {"metric": metric, "period_a": periods[0], "period_b": periods[1]},
                   compare_periods(metric, periods[0], periods[1]))
        if "error" in obs:
            return finish(f"Не удалось сравнить: {obs['error']}")
        return finish(
            f"{metric}: {obs['a']['value']} ({obs['a']['date']}) → {obs['b']['value']} ({obs['b']['date']}), "
            f"изменение в {obs['ratio']}× (delta {obs['delta']}).{note(obs)}"
        )

    # 2) Реальная ключевая ставка = ставка − инфляция
    if "реальн" in ql and "ставк" in ql:
        kr = emit("get_key_rate", {"on_date": None}, get_key_rate(None))
        lat = _latest(get_inflation, "cpi_yoy")
        inf = emit("get_inflation", {"year": lat[0], "month": lat[1]}, get_inflation(lat[0], lat[1]))
        expr = f"{kr['rate']} - {inf['cpi_yoy']}"
        c = emit("calculate", {"expression": expr}, calculate(expr))
        return finish(f"Реальная ключевая ставка ≈ {c['result']}% (номинальная {kr['rate']}% − инфляция {inf['cpi_yoy']}% г/г).{note(kr)}")

    # 3) Индекс нищеты = инфляция + безработица
    if "нищет" in ql or ("инфляц" in ql and "безработиц" in ql):
        li = _latest(get_inflation, "cpi_yoy")
        lu = _latest(get_unemployment, "unemployment")
        inf = emit("get_inflation", {"year": li[0], "month": li[1]}, get_inflation(li[0], li[1]))
        un = emit("get_unemployment", {"year": lu[0], "month": lu[1]}, get_unemployment(lu[0], lu[1]))
        expr = f"{inf['cpi_yoy']} + {un['unemployment']}"
        c = emit("calculate", {"expression": expr}, calculate(expr))
        return finish(f"Индекс нищеты ≈ {c['result']} (инфляция {inf['cpi_yoy']}% + безработица {un['unemployment']}%).")

    # 4) Правило 72 — за сколько лет удвоится вклад
    if "удво" in ql or "правил" in ql and "72" in ql or "72" in ql and "вклад" in ql:
        kr = emit("get_key_rate", {"on_date": None}, get_key_rate(None))
        expr = f"72 / {kr['rate']}"
        c = emit("calculate", {"expression": expr}, calculate(expr))
        return finish(f"При ставке {kr['rate']}% вклад удвоится примерно за {c['result']} лет (правило 72).{note(kr)}")

    # 5) Реальная доходность вклада
    if "доходност" in ql and "вклад" in ql:
        kr = emit("get_key_rate", {"on_date": None}, get_key_rate(None))
        lat = _latest(get_inflation, "cpi_yoy")
        inf = emit("get_inflation", {"year": lat[0], "month": lat[1]}, get_inflation(lat[0], lat[1]))
        expr = f"((1 + {kr['rate']}/100) / (1 + {inf['cpi_yoy']}/100) - 1) * 100"
        c = emit("calculate", {"expression": expr}, calculate(expr))
        return finish(f"Реальная доходность вклада ≈ {c['result']}% (ставка {kr['rate']}%, инфляция {inf['cpi_yoy']}%).{note(kr)}")

    # 6) Курс валюты на дату(ы)
    if cur and ("курс" in ql or "стоит" in ql or "стоил" in ql or "доллар" in ql or "евро" in ql or "юан" in ql):
        dates = periods or [None]
        if re.search(r"сегодня|сейчас", ql) and None not in dates:
            dates = dates + [None]
        vals = []
        for d in dates:
            on = d
            if isinstance(d, str) and re.fullmatch(r"\d{4}-\d{2}", d):
                on = d + "-01"
            obs = emit("get_fx_rate", {"currency": cur, "on_date": on}, get_fx_rate(cur, on))
            if "rate" in obs:
                vals.append((obs["date"], obs["rate"], obs.get("source", "")))
        if not vals:
            return finish(f"Не удалось получить курс {cur}.")
        parts = ", ".join(f"{v[1]} ₽ ({v[0]})" for v in vals)
        nt = " (часть значений из локального архива)" if any("fallback" in v[2] for v in vals) else ""
        return finish(f"Курс {cur}: {parts}.{nt}")

    # 7) Инфляция на конкретный месяц
    if ("инфляц" in ql or "ипц" in ql) and periods:
        y, m = map(int, (periods[0].split("-") + ["1"])[:2])
        obs = emit("get_inflation", {"year": y, "month": m}, get_inflation(y, m))
        if "error" in obs:
            return finish(f"Нет данных по инфляции: {obs['error']}")
        return finish(f"Инфляция (ИПЦ г/г) на {y}-{m:02d}: {obs['cpi_yoy']}%.")

    # 8) Безработица на месяц
    if "безработиц" in ql and periods:
        y, m = map(int, (periods[0].split("-") + ["1"])[:2])
        obs = emit("get_unemployment", {"year": y, "month": m}, get_unemployment(y, m))
        if "error" in obs:
            return finish(f"Нет данных по безработице: {obs['error']}")
        return finish(f"Безработица на {y}-{m:02d}: {obs['unemployment']}%.")

    # 9) Ключевая ставка (на дату или сейчас)
    if "ставк" in ql:
        d = periods[0] if periods else None
        obs = emit("get_key_rate", {"on_date": d}, get_key_rate(d))
        if "error" in obs:
            return finish(f"Нет данных по ставке: {obs['error']}")
        return finish(f"Ключевая ставка {'на ' + d if d else 'сейчас'}: {obs['rate']}%.{note(obs)}")

    return finish("Не удалось разобрать вопрос в офлайн-режиме (нет подходящего инструмента).")


if __name__ == "__main__":
    main()
