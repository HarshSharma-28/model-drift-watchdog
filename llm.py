"""LLM helper: local Ollama model with a template-based mock fallback.

The LLM only explains decisions in plain English. It never makes the
decision itself, which keeps every outcome auditable.
"""

OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "llama3.2:3b"


def ollama_available() -> bool:
    """Return True if a local Ollama server is reachable."""
    raise NotImplementedError


def explain(prompt: str, fallback: str) -> str:
    """Ask Ollama for an explanation, or return the template fallback."""
    raise NotImplementedError
