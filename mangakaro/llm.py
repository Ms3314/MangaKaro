"""Talking to a local Ollama server."""
import base64
import json
import urllib.error
import urllib.request

import cv2

DEFAULT_URL = "http://localhost:11434"
_url = DEFAULT_URL


class OllamaError(RuntimeError):
    pass


class OllamaTimeout(OllamaError):
    """The model did not answer in time (a stalled generation). Callers can retry with a smaller request."""


def configure(url=DEFAULT_URL):
    global _url
    _url = url.rstrip("/")


def b64_jpg(img):
    return base64.b64encode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()).decode()


def chat(model, prompt, schema, images=(), think=False, timeout=180, max_tokens=2048):
    """One structured-output call. `schema` is a JSON schema the reply must follow. Returns the reply text."""
    body = {"model": model, "stream": False, "format": schema, "think": think, "keep_alive": "30m",
            "options": {"temperature": 0, "num_ctx": 8192, "num_predict": max_tokens},  # num_predict stops a looping generation
            "messages": [{"role": "user", "content": prompt, **({"images": [b64_jpg(i) for i in images]} if images else {})}]}
    req = urllib.request.Request(f"{_url}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)["message"]["content"]
    except TimeoutError as e:
        raise OllamaTimeout(f"model {model} did not answer within {timeout}s") from e
    except urllib.error.HTTPError as e:
        raise OllamaError(f"Ollama returned {e.code} for model {model}: {e.read().decode()[:300]}") from e
    except urllib.error.URLError as e:
        raise OllamaError(f"cannot reach Ollama at {_url} ({e.reason}). Is `ollama serve` running?") from e


def installed_models():
    """Names of the models the server has, or raise OllamaError if it is not reachable."""
    try:
        with urllib.request.urlopen(f"{_url}/api/tags", timeout=10) as r:
            return [m["name"] for m in json.load(r)["models"]]
    except (urllib.error.URLError, OSError) as e:
        raise OllamaError(f"cannot reach Ollama at {_url} ({e}). Is `ollama serve` running?") from e


def require_models(models):
    """Raise OllamaError naming every model that is not installed, with the pull command for each."""
    have = set(installed_models())
    have |= {n.removesuffix(":latest") for n in have}
    missing = [m for m in dict.fromkeys(models) if m not in have and f"{m}:latest" not in have]
    if missing:
        raise OllamaError("model not installed in Ollama: " + ", ".join(missing) + "\n" + "\n".join(f"  ollama pull {m}" for m in missing))
