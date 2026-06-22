"""
make_error_traces.py — прогнать три РАЗНЫХ типа ошибок через настоящий
исполнитель агента _exec_one и дописать их в trace.jsonl (для раздела
«Диагностика»). Это те же ветки обработки ошибок, что срабатывают при живой
LLM, только триггерим их детерминированно (офлайн модель сама битый вызов
не сгенерирует).

    python make_error_traces.py
"""

from __future__ import annotations

import json
import uuid
from types import SimpleNamespace

from agent import _exec_one, _now_iso, append_jsonl


def mock_call(name: str, arguments: str):
    return SimpleNamespace(
        id="call_" + name,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


CASES = [
    # 1) Битый JSON в аргументах (модель оборвала строку аргументов)
    ("битый JSON в аргументах", mock_call("get_fx_rate", '{"currency": "USD"')),
    # 2) Галлюцинация инструмента (модель позвала несуществующий get_gdp)
    ("галлюцинация инструмента", mock_call("get_gdp", '{"year": 2025}')),
    # 3) Инструмент упал на неверных аргументах (лишний kwarg currency у get_key_rate)
    ("инструмент упал: плохие аргументы", mock_call("get_key_rate", '{"currency": "USD"}')),
]


def main():
    run_id = "errcases_" + uuid.uuid4().hex[:6]
    for i, (label, tc) in enumerate(CASES, 1):
        _, args, obs = _exec_one(tc)
        append_jsonl({
            "run_id": run_id, "ts": _now_iso(), "step": i,
            "call": tc.function.name, "args": args, "obs": obs, "error_type": label,
        })
        print(f"[{label}] {tc.function.name} -> {json.dumps(obs, ensure_ascii=False)}")
    print(f"\nДописано в trace.jsonl (run_id={run_id})")


if __name__ == "__main__":
    main()
