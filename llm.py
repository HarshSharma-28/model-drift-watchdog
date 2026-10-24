"""LLM layer: real, free LLMs with a safe fallback chain.

Provider order (first one that works is used):
  1. Groq API      — free tier, runs Llama 3 in the cloud (works on Streamlit Cloud).
                      Needs GROQ_API_KEY (env var, Streamlit secret, or typed in the sidebar).
  2. Ollama        — free local model on your own PC (OLLAMA_URL / OLLAMA_MODEL).
  3. Template mock — only if no LLM is reachable, so the app never breaks.

The LLM writes explanations, incident summaries and chat answers. It never makes or
changes a decision: decisions come from `policy.py` and are validated by the Critic.
"""

from __future__ import annotations

import os
import time

import requests

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_FALLBACK_MODEL = "llama-3.1-8b-instant"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2:3b")

SYSTEM_PROMPT = (
    "You are the Reporter agent of 'Model Drift Watchdog', an agentic MLOps system that monitors a credit-risk "
    "model at a digital lender. Write for a business stakeholder in plain English. Use ONLY the facts given; "
    "never invent numbers. The decision was made by a fixed, audited policy and validated by a Critic agent - "
    "explain it, do not change or question it. Cite knowledge-base document IDs in square brackets when you use them."
)

_STATUS_CACHE: dict = {"t": 0.0, "value": None}


def groq_key() -> str | None:
    return os.environ.get("GROQ_API_KEY") or None


def groq_model() -> str:
    return os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


def ollama_status(timeout: float = 1.5) -> dict:
    """Is a local Ollama server reachable, and does it have the model?"""
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


def llm_status(force: bool = False) -> dict:
    """Which real LLM will be used right now: {'provider', 'model', 'label', 'detail'}."""
    now = time.time()
    if not force and _STATUS_CACHE["value"] and now - _STATUS_CACHE["t"] < 30:
        return _STATUS_CACHE["value"]
    if groq_key():
        value = {"provider": "groq", "model": groq_model(), "label": f"Groq · {groq_model()}",
                 "detail": "Cloud LLM (free tier)"}
    else:
        o = ollama_status()
        if o["available"]:
            value = {"provider": "ollama", "model": OLLAMA_MODEL, "label": f"Ollama · {OLLAMA_MODEL}",
                     "detail": "Local LLM on this computer"}
        else:
            value = {"provider": "mock", "model": None, "label": "No LLM connected",
                     "detail": ("Ollama is running but the model isn't pulled" if o["server"] else
                                "Add a Groq API key or start Ollama")}
    _STATUS_CACHE.update(t=now, value=value)
    return value


def _groq_chat(messages: list[dict], max_tokens: int, temperature: float, timeout: float) -> tuple[str, str] | None:
    key = groq_key()
    if not key:
        return None
    for model in dict.fromkeys([groq_model(), GROQ_FALLBACK_MODEL]):
        try:
            r = requests.post(GROQ_URL, headers={"Authorization": f"Bearer {key}"}, timeout=timeout,
                              json={"model": model, "messages": messages, "max_tokens": max_tokens,
                                    "temperature": temperature})
            if r.status_code == 200:
                text = r.json()["choices"][0]["message"]["content"].strip()
                if text:
                    return text, f"groq:{model}"
        except Exception:
            continue
    return None


def _ollama_chat(messages: list[dict], max_tokens: int, temperature: float, timeout: float) -> tuple[str, str] | None:
    if not ollama_available():
        return None
    try:
        r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=timeout,
                          json={"model": OLLAMA_MODEL, "messages": messages, "stream": False,
                                "options": {"temperature": temperature, "num_predict": max_tokens}})
        r.raise_for_status()
        text = r.json().get("message", {}).get("content", "").strip()
        if text:
            return text, f"ollama:{OLLAMA_MODEL}"
    except Exception:
        pass
    return None


def chat(messages: list[dict], max_tokens: int = 400, temperature: float = 0.2,
         timeout: float = 45.0) -> tuple[str, str] | None:
    """Send chat messages to the first available real LLM. Returns (text, source) or None."""
    return (_groq_chat(messages, max_tokens, temperature, timeout)
            or _ollama_chat(messages, max_tokens, temperature, timeout))


def explain(facts: str, fallback: str, use_llm: bool = True, timeout: float = 45.0) -> tuple[str, str]:
    """Incident summary for the report. Returns (text, source); source is 'groq:…', 'ollama:…' or 'mock'."""
    if use_llm:
        result = chat([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Monitoring facts:\n{facts}\n\n"
                "Write a short incident summary with exactly these three bold headings, 1-3 sentences each:\n"
                "**What happened** - **Why it matters** - **What we are doing** (follow the runbook steps given). "
                "Mention the key numbers from the facts.")},
        ], max_tokens=350, timeout=timeout)
        if result:
            return result
    return fallback, "mock"


def source_label(source: str) -> str:
    if source.startswith("groq:"):
        return "Groq · " + source.split(":", 1)[1]
    if source.startswith("ollama:"):
        return "Ollama · " + source.split(":", 1)[1]
    return "Template (no LLM connected)"
