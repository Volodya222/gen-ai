"""
Семинар 1 — Часть 3: Эксперимент с temperature
================================================
Один промпт, несколько запусков при разных параметрах креативности.
Цель: увидеть разницу между детерминированным и стохастическим режимами.
"""

import os
import time

from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════
# Выберите провайдера (раскомментируйте один)
# ══════════════════════════════════════════

PROVIDER = "openai"  # ← вариант 1 (OpenAI-совместимый)
# PROVIDER = "gemini"     # ← вариант 2 (нужен GEMINI_API_KEY)

# ══════════════════════════════════════════

PROMPT = (
    "Назови одним словом главную проблему российской экономики. Ответь одним словом."
)
N_RUNS = 10


def run_openai(temperature: float, n: int) -> list[str]:
    from llm_client import get_model, make_raw_client

    client = make_raw_client()
    model = get_model()
    results = []
    for i in range(n):
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=temperature,
            max_tokens=50,
        )
        results.append(resp.choices[0].message.content.strip())
        time.sleep(0.3)  # пауза, чтобы не флудить
    return results


def run_gemini(temperature: float, n: int) -> list[str]:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    results = []
    for i in range(n):
        resp = client.models.generate_content(
            model=os.getenv("LLM_MODEL"),
            contents=PROMPT,
            config={"temperature": temperature, "max_output_tokens": 50},
        )
        results.append(resp.text.strip())
        time.sleep(0.3)
    return results


def analyze(results: list[str], label: str):
    unique = set(results)
    print(f"\n{'═' * 50}")
    print(f"  {label}")
    print(f"{'═' * 50}")
    for i, r in enumerate(results, 1):
        print(f"  [{i:2d}] {r}")
    print(f"\n  Уникальных ответов: {len(unique)} из {len(results)}")
    print(f"  Процент уникальных: {len(unique) / len(results) * 100:.0f}%")


if __name__ == "__main__":
    run = run_openai if PROVIDER == "openai" else run_gemini
    print(f"Провайдер: {PROVIDER}")
    print(f'Промпт: "{PROMPT}"')
    print(f"Прогонов: {N_RUNS}")

    # Эксперимент 1: temperature = 0
    results_t0 = run(temperature=0, n=N_RUNS)
    analyze(results_t0, "Temperature = 0 (детерминированный)")

    # Эксперимент 2: temperature = 1.0
    results_t1 = run(temperature=1.0, n=N_RUNS)
    analyze(results_t1, "Temperature = 1.0 (стохастический)")

    # ─── Визуализация ───
    print(f"\n{'═' * 50}")
    print("  Визуализация")
    print(f"{'═' * 50}")

    try:
        import matplotlib.pyplot as plt

        labels = ["T=0", "T=1.0"]
        unique_pct = [
            len(set(results_t0)) / len(results_t0) * 100,
            len(set(results_t1)) / len(results_t1) * 100,
        ]
        colors = ["#4A90D9", "#F96167"]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(labels, unique_pct, color=colors, width=0.5)
        ax.set_ylabel("% уникальных ответов")
        ax.set_title(f"Влияние temperature ({N_RUNS} прогонов)")
        ax.set_ylim(0, 110)
        for bar, pct in zip(bars, unique_pct):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 3,
                f"{pct:.0f}%",
                ha="center",
                fontsize=14,
                fontweight="bold",
            )
        plt.tight_layout()
        plt.savefig("temperature_experiment.png", dpi=150)
        print("  График сохранён: temperature_experiment.png")
        plt.show()
    except Exception as e:
        print(f"  Не удалось построить график: {e}")
