"""
Семинар 1 — Часть 4: Оценщик заголовков
=========================================
Читаем CSV с заголовками, отправляем каждый в LLM,
получаем оценку кликбейтности, записываем результат.
"""

import os
import time

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# ══════════════════════════════════════════
# Выберите провайдера (раскомментируйте один)
# ══════════════════════════════════════════

PROVIDER = "openai"  # ← вариант 1
# PROVIDER = "gemini"     # ← вариант 2

# ══════════════════════════════════════════

SYSTEM_PROMPT = """Ты — эксперт по медиа и журналистике.
Оцени кликбейтность заголовка новости по шкале от 1 до 10, где:
1 — полностью нейтральный, информативный заголовок
10 — максимальный кликбейт (капс, манипуляция, ложная сенсация)

Ответь ТОЛЬКО числом от 1 до 10, без пояснений."""

N_RUNS = 5  # сколько раз оценить каждый заголовок


def score_openai(headline: str) -> int:
    from llm_client import get_model, make_raw_client

    client = make_raw_client()
    model = get_model()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Заголовок: {headline}"},
        ],
        temperature=0,
        max_tokens=5,
    )
    text = resp.choices[0].message.content.strip()
    try:
        return int(text)
    except ValueError:
        # иногда модель добавляет пояснение — берём первое число
        import re

        m = re.search(r"\d+", text)
        return int(m.group()) if m else -1


def score_gemini(headline: str) -> int:
    from google import genai

    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    resp = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=f"Заголовок: {headline}",
        config={
            "system_instruction": SYSTEM_PROMPT,
            "temperature": 0,
            "max_output_tokens": 5,
        },
    )
    text = resp.text.strip()
    try:
        return int(text)
    except ValueError:
        import re

        m = re.search(r"\d+", text)
        return int(m.group()) if m else -1


def main():
    score_fn = score_openai if PROVIDER == "openai" else score_gemini
    print(f"Провайдер: {PROVIDER}")
    print(f"Прогонов на заголовок: {N_RUNS}\n")

    # Читаем CSV
    df = pd.read_csv("headlines.csv")
    print(f"Загружено {len(df)} заголовков\n")

    if N_RUNS == 1:
        # Простой режим: одна оценка
        scores = []
        for _, row in df.iterrows():
            s = score_fn(row["headline"])
            scores.append(s)
            print(f"  [{s:2d}] {row['headline'][:60]}")
            time.sleep(0.3)
        df["score"] = scores
    else:
        # Статистический режим: N оценок → среднее и std
        all_scores = {i: [] for i in df.index}
        for run in range(N_RUNS):
            print(f"  Прогон {run + 1}/{N_RUNS}...")
            for idx, row in df.iterrows():
                s = score_fn(row["headline"])
                all_scores[idx].append(s)
                time.sleep(0.3)

        df["score_mean"] = [sum(v) / len(v) for v in all_scores.values()]
        df["score_std"] = [
            (sum((x - sum(v) / len(v)) ** 2 for x in v) / len(v)) ** 0.5
            for v in all_scores.values()
        ]
        df["scores_raw"] = [str(v) for v in all_scores.values()]

        print()
        for _, row in df.iterrows():
            print(
                f"  [{row['score_mean']:.1f} ± {row['score_std']:.1f}] {row['headline'][:55]}"
            )

    # Сохраняем результат
    out_file = "headlines_scored.csv"
    df.to_csv(out_file, index=False)
    print(f"\nРезультаты сохранены в {out_file}")

    # ─── Дискуссия: что мы только что увидели ───
    if "score_std" in df.columns:
        df_sorted = df.sort_values("score_std", ascending=False)
        print("\n" + "═" * 60)
        print("СТАБИЛЬНОСТЬ ОЦЕНОК — какие заголовки модель оценила непостоянно")
        print("═" * 60)
        print("\nТоп-3 самых нестабильных (std оценки выше всего):")
        for _, row in df_sorted.head(3).iterrows():
            print(
                f"  ± {row['score_std']:.2f}  [среднее {row['score_mean']:.1f}]  "
                f"{row['headline'][:50]}"
            )
        print("\nТоп-3 самых стабильных (модель всегда одинаково):")
        for _, row in df_sorted.tail(3).iterrows():
            print(
                f"  ± {row['score_std']:.2f}  [среднее {row['score_mean']:.1f}]  "
                f"{row['headline'][:50]}"
            )
        print(
            "\n💡 Обсуждение в группе:\n"
            "   • temperature=0, а оценки всё равно плавают — почему?\n"
            "   • какие заголовки нестабильны: явный кликбейт или серые?\n"
            "   • можем ли мы доверять одной оценке как «истинной»?\n"
        )

    # ─── Визуализация ───
    try:
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams["font.family"] = "DejaVu Sans"

        score_col = "score" if "score" in df.columns else "score_mean"
        std_col = "score_std" if "score_std" in df.columns else None

        fig, ax = plt.subplots(figsize=(10, 6))
        colors = [
            "#4A90D9" if s <= 4 else "#F9A825" if s <= 7 else "#F96167"
            for s in df[score_col]
        ]
        if std_col:
            bars = ax.barh(
                range(len(df)),
                df[score_col],
                xerr=df[std_col],
                color=colors,
                error_kw={"ecolor": "#333", "capsize": 3, "lw": 1},
            )
            title = f"Кликбейтность ({N_RUNS} прогонов, среднее ± std)"
        else:
            bars = ax.barh(range(len(df)), df[score_col], color=colors)
            title = "Оценка кликбейтности заголовков"
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(
            [h[:45] + "..." if len(h) > 45 else h for h in df["headline"]], fontsize=8
        )
        ax.set_xlabel("Кликбейтность (1–10)")
        ax.set_title(title)
        ax.set_xlim(0, 11)
        ax.invert_yaxis()
        plt.tight_layout()
        plt.savefig("headline_scores.png", dpi=150, bbox_inches="tight")
        print("График сохранён: headline_scores.png")
        plt.show()
    except Exception as e:
        print(f"Не удалось построить график: {e}")

    # ─── переход к семинару 2 ───
    print("\n" + "═" * 60)
    print("ЧТО ДАЛЬШЕ")
    print("═" * 60)
    print(
        "Заметили? Чтобы достать число из ответа модели, мы здесь руками\n"
        "ищем регулярку через re.search(r'\\d+', text). Это работает, но\n"
        "криво: модель иногда добавит пояснение, иногда вернёт «семь»\n"
        "словом, и наш парсер упадёт.\n\n"
        "На следующем занятии исправим это раз и навсегда: опишем\n"
        "ожидаемый ответ как Pydantic-класс и заставим API возвращать\n"
        "сразу типизированный объект. Без регулярок, без try/except,\n"
        "без догадок «что там модель имела в виду».\n"
    )


if __name__ == "__main__":
    main()
