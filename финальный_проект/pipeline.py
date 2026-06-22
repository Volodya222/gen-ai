"""
Ядро проекта: один раунд игры «Ультиматум» как мульти-агентный пайплайн.

Техники курса, реально работающие здесь:
  1. Синтетические персоны  — Отвечающий обусловлен демографическим профилем.
  2. Агент с инструментами   — Предлагающий зовёт калькулятор выплат (tools.py)
                               перед ходом; числа берутся из инструмента.
  3. Мультиагент             — Предлагающий и Отвечающий взаимодействуют, ход
                               одного становится входом другого.
  4. LLM-as-judge            — отдельная модель оценивает согласованность хода
                               Отвечающего.
  5. Структурированный вывод — все ходы через response_model + field_validator.
  6. Проверка галлюцинаций   — детектор ghost-раундов и выдуманных чисел.

Один раунд: offer (задан в eval или выбран агентом) → Responder решает →
settle выплаты инструментом → judge оценивает → детектор галлюцинаций.
"""
from __future__ import annotations

from dataclasses import dataclass

from llm_client import get_model, make_client
from personas import Persona
from prompts import JUDGE_SYSTEM, PROPOSER_SYSTEM, RESPONDER_SYSTEM_TMPL
from schema import JudgeVerdict, ProposerMove, ResponderMove
from tools import acceptance_stats, expected_payoffs, settle_round

PIE = 100

_client = None
_model = None


def _c():
    global _client, _model
    if _client is None:
        _client = make_client()
        _model = get_model()
    return _client, _model


# ---------------------------------------------------------------------------
# Трейс одного раунда — фиксируем путь (для eval: шаги, инструменты, токены)
# ---------------------------------------------------------------------------


@dataclass
class RoundTrace:
    round: int
    persona_id: str
    offer: int
    steps: list[str]            # последовательность шагов пайплайна
    tools_called: list[str]     # какие инструменты вызваны
    accept: bool
    fairness_perceived: str
    proposer_payoff: int
    responder_payoff: int
    judge_score: int
    judge_consistent: bool
    hallucinations: list[str]   # пойманные галлюцинации (ghost-раунды/числа)
    prompt_tokens: int
    completion_tokens: int


# ---------------------------------------------------------------------------
# Агент-Предлагающий с инструментом (используется в свободном режиме)
# ---------------------------------------------------------------------------


def proposer_decide(round_num: int, history: list[dict]) -> tuple[ProposerMove, list[str], dict]:
    """
    Агент сначала зовёт инструмент expected_payoffs для нескольких кандидатов
    (грубый перебор), затем просит модель выбрать предложение, опираясь на числа.
    Возвращает (ход, список вызванных инструментов, usage).
    """
    client, model = _c()
    tools_called = []

    # ReAct-шаг: оценим вероятность приёма по истории инструментом.
    stats = acceptance_stats(history)
    tools_called.append("acceptance_stats")
    base_p = stats["accept_rate"] if stats["n"] >= 2 else 0.6

    # Прогоним кандидатов через калькулятор (инструмент даёт реальные EV).
    candidates = {}
    for off in (10, 20, 30, 40, 50):
        # эвристика: чем щедрее, тем выше шанс приёма
        p = min(1.0, base_p + (off - 30) / 100)
        candidates[off] = expected_payoffs(off, max(0.0, p))
    tools_called.append("expected_payoffs")

    calc_text = "\n".join(
        f"  отдать {off}: мой EV={c['ev_proposer']}, его EV={c['ev_responder']}"
        for off, c in candidates.items()
    )
    past = "\n".join(
        f"  раунд {h['round']}: предложил {h['offer']} → {'принято' if h['accept'] else 'отказ'}"
        for h in history
    ) or "  (первый раунд)"

    move = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": PROPOSER_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Раунд {round_num}. История:\n{past}\n\n"
                    f"Расчёт ожидаемых выплат (инструмент):\n{calc_text}\n\n"
                    "Сколько отдашь Отвечающему?"
                ),
            },
        ],
        response_model=ProposerMove,
        temperature=0.7,
        max_retries=2,
    )
    usage = _last_usage(client)
    return move, tools_called, usage


# ---------------------------------------------------------------------------
# Агент-Отвечающий (синтетическая персона)
# ---------------------------------------------------------------------------


def responder_decide(
    persona: Persona, offer: int, round_num: int, history: list[dict]
) -> tuple[ResponderMove, dict]:
    client, model = _c()
    stance = "справедливость" + (
        " важнее денег" if persona.reject_below >= 35 else " важна, но деньги важнее"
    )
    sys = RESPONDER_SYSTEM_TMPL.format(profile=persona.profile, stance=stance)
    past = "\n".join(
        f"  раунд {h['round']}: тебе предложили {h['offer']} → ты {'принял' if h['accept'] else 'отверг'}"
        for h in history
    ) or "  (первый раунд)"

    move = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": sys},
            {
                "role": "user",
                "content": (
                    f"Раунд {round_num}. История:\n{past}\n\n"
                    f"Тебе предлагают {offer} из 100 ₽. Принять или отвергнуть?"
                ),
            },
        ],
        response_model=ResponderMove,
        temperature=0.7,
        max_retries=2,
    )
    return move, _last_usage(client)


# ---------------------------------------------------------------------------
# Судья (LLM-as-judge)
# ---------------------------------------------------------------------------


def judge_move(offer: int, move: ResponderMove) -> tuple[JudgeVerdict, dict]:
    client, model = _c()
    verdict = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {
                "role": "user",
                "content": (
                    f"Предложение Отвечающему: {offer} из 100 ₽.\n"
                    f"Решение: {'принял' if move.accept else 'отверг'}.\n"
                    f"Воспринятая справедливость: {move.fairness_perceived}.\n"
                    f"Обоснование: {move.reasoning!r}\n\n"
                    "Оцени согласованность и человекоподобность хода."
                ),
            },
        ],
        response_model=JudgeVerdict,
        temperature=0.0,
        max_retries=2,
    )
    return verdict, _last_usage(client)


# ---------------------------------------------------------------------------
# Детектор галлюцинаций
# ---------------------------------------------------------------------------


def detect_hallucinations(
    move: ResponderMove | ProposerMove, round_num: int, offer: int
) -> list[str]:
    """
    Ловим два класса галлюцинаций:
      * ghost-раунд: ссылка на раунд, которого ещё не было (>= текущего).
      * выдуманное число: в reasoning встречается денежная сумма, не равная
        offer и не входящая в множество допустимых ориентиров (0, 50, 100, keep).
    """
    flags: list[str] = []
    if move.refers_to_round is not None and move.refers_to_round >= round_num:
        flags.append(
            f"ghost_round: ссылка на раунд {move.refers_to_round} в раунде {round_num}"
        )

    import re

    nums = {int(n) for n in re.findall(r"\b(\d{1,3})\b", move.reasoning)}
    allowed = {offer, PIE - offer, 0, 50, 100}
    ghost_nums = {n for n in nums if n <= 100 and n not in allowed}
    # эвристика: денежные числа в reasoning, не связанные с этим раундом
    if ghost_nums:
        flags.append(f"ghost_number: упомянуты числа {sorted(ghost_nums)} не из раунда")
    return flags


# ---------------------------------------------------------------------------
# Один раунд целиком
# ---------------------------------------------------------------------------


def run_round(
    persona: Persona, offer: int, round_num: int, history: list[dict]
) -> RoundTrace:
    steps = ["persona_conditioning"]
    tools_called: list[str] = []

    move, r_usage = responder_decide(persona, offer, round_num, history)
    steps.append("responder_decide")

    settled = settle_round(offer, move.accept)
    tools_called.append("settle_round")
    steps.append("settle")

    verdict, j_usage = judge_move(offer, move)
    steps.append("judge")

    halluc = detect_hallucinations(move, round_num, offer)
    steps.append("hallucination_check")

    return RoundTrace(
        round=round_num,
        persona_id=persona.id,
        offer=offer,
        steps=steps,
        tools_called=tools_called,
        accept=move.accept,
        fairness_perceived=move.fairness_perceived,
        proposer_payoff=settled["proposer_payoff"],
        responder_payoff=settled["responder_payoff"],
        judge_score=verdict.rationality_score,
        judge_consistent=verdict.consistent,
        hallucinations=halluc,
        prompt_tokens=r_usage["prompt"] + j_usage["prompt"],
        completion_tokens=r_usage["completion"] + j_usage["completion"],
    )


# ---------------------------------------------------------------------------
# Доступ к usage последнего вызова (токены — для метрики «путь/стоимость»)
# ---------------------------------------------------------------------------


def _last_usage(client) -> dict:
    """
    Достаём usage последнего вызова. JsonClient хранит его на внутреннем
    OpenAI-клиенте (client._c._last_usage). Если недоступно — нули.
    """
    try:
        inner = getattr(client, "_c", None)
        last = getattr(inner, "_last_usage", None)
        if last:
            return last
    except Exception:
        pass
    return {"prompt": 0, "completion": 0}
