"""Optional OpenAI backend, used only for the cloud comparison (experiments/openai_compare.py).

A model name starting with `openai:` (for example `openai:gpt-5.4-mini`) sends that stage's call to the OpenAI API instead of Ollama.
The key is read from the OPENAI_API_KEY environment variable and is never written anywhere. Every call is added to a ledger
with its token counts and cost, and a call is refused once the spending cap is reached."""
import os
import time

from .llm import OllamaError, OllamaTimeout, b64_jpg

PREFIX = "openai:"

# USD per 1M tokens (input, output). Copied from OpenAI's pricing page on 2026-10-19: check them before trusting a cost figure.
PRICES = {
    "gpt-6.1-sol": (2.00, 10.00),
    "gpt-6-luna": (0.10, 0.50),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-nano": (0.20, 1.25),
    "gpt-4.1-mini": (0.40, 1.60),
}
# image models: USD per 1M tokens (text in, image in, image out)
IMAGE_PRICES = {"gpt-image-2": (5.00, 8.00, 30.00), "gpt-image-1.5": (5.00, 8.00, 32.00), "gpt-image-1-mini": (2.00, 2.50, 8.00)}


class BudgetExceeded(RuntimeError):
    pass


class Ledger:
    def __init__(self, max_usd):
        self.max_usd, self.calls, self.arm = max_usd, [], "unassigned"

    @property
    def spent(self):
        return sum(c["usd"] for c in self.calls)

    def check(self):
        if self.spent >= self.max_usd:
            raise BudgetExceeded(f"spending cap reached: ${self.spent:.3f} of ${self.max_usd:.2f}")

    def add(self, model, tokens_in, tokens_out, usd, seconds):
        self.calls.append({"arm": self.arm, "model": model, "in": tokens_in, "out": tokens_out, "usd": round(usd, 5), "seconds": round(seconds, 1)})

    def by_arm(self):
        out = {}
        for c in self.calls:
            a = out.setdefault(c["arm"], {"calls": 0, "in": 0, "out": 0, "usd": 0.0, "seconds": 0.0})
            a["calls"] += 1
            a["in"] += c["in"]
            a["out"] += c["out"]
            a["usd"] = round(a["usd"] + c["usd"], 5)
            a["seconds"] = round(a["seconds"] + c["seconds"], 1)
        return out


LEDGER = Ledger(max_usd=2.0)
_client = None


def client():
    global _client
    if _client is None:
        try:
            import openai
        except ImportError as e:
            raise OllamaError("the openai package is not installed: uv pip install --python .venv/bin/python openai") from e
        if not os.environ.get("OPENAI_API_KEY"):
            raise OllamaError("OPENAI_API_KEY is not set in the environment")
        _client = openai.OpenAI(timeout=180, max_retries=1)
    return _client


def chat(model, prompt, schema, images=(), max_tokens=2048):
    """Same contract as llm.chat for Ollama: one structured-output call, returns the reply text."""
    import openai
    name = model.removeprefix(PREFIX)
    if name not in PRICES:
        raise OllamaError(f"no price known for {name}; add it to openai_backend.PRICES so the cost can be counted")
    LEDGER.check()
    content = [{"type": "text", "text": prompt}] + [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64_jpg(i)}} for i in images]
    t = time.perf_counter()
    try:
        r = client().chat.completions.create(
            model=name, messages=[{"role": "user", "content": content}],
            response_format={"type": "json_schema", "json_schema": {"name": "out", "schema": schema, "strict": False}},
            max_completion_tokens=max(max_tokens, 4096))  # reasoning models spend part of this on hidden thinking
    except openai.APITimeoutError as e:
        raise OllamaTimeout(f"{name} did not answer in time") from e
    except openai.OpenAIError as e:
        raise OllamaError(f"OpenAI error for {name}: {e}") from e
    pin, pout = PRICES[name]
    u = r.usage
    LEDGER.add(name, u.prompt_tokens, u.completion_tokens, (u.prompt_tokens * pin + u.completion_tokens * pout) / 1e6, time.perf_counter() - t)
    return r.choices[0].message.content or ""
