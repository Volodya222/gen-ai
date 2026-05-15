import os
import time

from dotenv import load_dotenv

load_dotenv()


PROMPT = (
    "Назови одним словом главную проблему российской экономики. Ответь одним словом."
)

from llm_client import get_model, make_raw_client

client = make_raw_client()
model = get_model()
temperature = 0.0
resp = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": PROMPT}],
    temperature=temperature,
    max_tokens=50,
)
resp.choices[0]
