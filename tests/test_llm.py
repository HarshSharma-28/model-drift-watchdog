"""LLM provider chain: Groq -> Ollama -> template fallback (network mocked)."""

import llm


class _Resp:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def _no_ollama(*a, **k):
    raise ConnectionError("no ollama")


def test_falls_back_to_template_without_any_llm(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm.requests, "get", _no_ollama)
    assert llm.llm_status(force=True)["provider"] == "mock"
    assert llm.explain("facts", "fallback text") == ("fallback text", "mock")


def test_uses_groq_when_key_present(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append((url, json["model"]))
        return _Resp(200, {"choices": [{"message": {"content": "**What happened** drift."}}]})

    monkeypatch.setattr(llm.requests, "post", fake_post)
    assert llm.llm_status(force=True)["provider"] == "groq"
    text, source = llm.explain("facts", "fallback")
    assert text.startswith("**What happened**") and source.startswith("groq:")
    assert calls[0][0] == llm.GROQ_URL


def test_groq_falls_back_to_smaller_model(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(llm.requests, "post", lambda url, headers=None, json=None, timeout=None:
                        _Resp(404, {}) if json["model"] != llm.GROQ_FALLBACK_MODEL
                        else _Resp(200, {"choices": [{"message": {"content": "ok"}}]}))
    assert llm.chat([{"role": "user", "content": "hi"}]) == ("ok", f"groq:{llm.GROQ_FALLBACK_MODEL}")


def test_uses_ollama_when_no_groq(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr(llm.requests, "get", lambda *a, **k: _Resp(200, {"models": [{"name": llm.OLLAMA_MODEL}]}))
    monkeypatch.setattr(llm.requests, "post", lambda *a, **k: _Resp(200, {"message": {"content": "local answer"}}))
    assert llm.llm_status(force=True)["provider"] == "ollama"
    assert llm.chat([{"role": "user", "content": "hi"}]) == ("local answer", f"ollama:{llm.OLLAMA_MODEL}")
