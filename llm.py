"""LLM helper: a free local Ollama model, with a template-based mock fallback.

The LLM only *explains* decisions in plain English. It never makes or
changes a decision, which keeps every outcome auditable and repeatable.
If Ollama isn't installed or running, the mock fallback is used, so the
demo works on any laptop with zero setup.

Configure with environment variables:
    OLLAMA_URL    (default http://localhost:11434)
    OLLAMA_MODEL  (default llama3.2:3b)
"""

from __future__ import annotations

import os

import requests

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = (
    "You are the Reporter agent in an ML monitoring system for a lending company. "
    "Explain the monitoring result to a business stakeholder in 3-4 short sentences of plain English. "
    "Use ONLY the facts given. Do not invent numbers. Do not change or question the decision; "
    "it was made by a fixed, audited policy and validated by a Critic agent."
)


def ollama_status(timeout: float = 1.5) -> dict:
    """Check whether a local Ollama server is reachable and has the model."""
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=timeout)
        r.raise_for_status()
        models = [m.get("name", "") for m in r.json().get("models", [])]
        has_model = any(m == OLLAMA_MODEL or m.split(":")[0] == OLLAMA_MODEL.split(":")[0] for m in models)
        return {"available": has_model, "server": True, "models": models}
    except Exception:
        return {"available": False, "server": False, "models": []}


def ollama_available() -> bool:
    return ollama_status()["available"]


def explain(facts: str, fallback: str, use_llm: bool = True, timeout: float = 60.0) -> tuple[str, str]:
    """Return (explanation, source). Source is 'ollama:<model>' or 'mock'."""
    if use_llm and ollama_available():
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/generate",
                json={
                    "model": OLLAMA_MODEL,
                    "system": SYSTEM_PROMPT,
                    "prompt": f"Monitoring facts:\n{facts}\n\nWrite the explanation now.",
                    "stream": False,
                    "options": {"temperature": 0.2, "num_predict": 220},
                },
                timeout=timeout,
            )
            r.raise_for_status()
            text = r.json().get("response", "").strip()
            if text:
                return text, f"ollama:{OLLAMA_MODEL}"
        except Exception:
            pass
    return fallback, "mock"
