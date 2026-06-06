"""Multi-provider fallback engine (MiniMax -> Gemini -> local Qwen).

Verifies ordering, fall-through on failure, the offline-local guarantee, and
that an explicit (utility) model goes straight to local. Providers are faked, so
no network or `openai` package is needed.
"""

from __future__ import annotations

from llm.engine import FallbackLLM


class FakeProvider:
    def __init__(self, name, behavior):
        # behavior: ("ok", text) | ("raise",) | ("empty",)
        self.name = name
        self.behavior = behavior
        self.calls = []

    def complete(self, system, messages, *, model, max_tokens, temperature):
        self.calls.append(model)
        kind = self.behavior[0]
        if kind == "raise":
            raise RuntimeError(f"{self.name} down")
        if kind == "empty":
            return ""
        return self.behavior[1]


def _engine(*providers):
    eng = FallbackLLM()
    eng._providers = list(providers)  # bypass _build (no settings/network)
    return eng


def _call(eng, model=None):
    return eng.complete("sys", [{"role": "user", "content": "q"}],
                        model=model, max_tokens=64, temperature=0.2)


def test_uses_first_working_provider():
    p1 = FakeProvider("minimax", ("ok", "from minimax"))
    p2 = FakeProvider("gemini", ("ok", "from gemini"))
    eng = _engine(p1, p2)
    assert _call(eng) == "from minimax"
    assert p2.calls == []  # second provider never tried


def test_falls_through_failures_to_next():
    p1 = FakeProvider("minimax", ("raise",))
    p2 = FakeProvider("gemini", ("empty",))
    p3 = FakeProvider("local", ("ok", "from local"))
    eng = _engine(p1, p2, p3)
    assert _call(eng) == "from local"
    assert p1.calls and p2.calls and p3.calls  # all attempted in order


def test_offline_local_guarantee_when_cloud_down():
    cloud = FakeProvider("gemini", ("raise",))
    local = FakeProvider("local-ollama", ("ok", "offline answer"))
    assert _call(_engine(cloud, local)) == "offline answer"


def test_explicit_model_goes_straight_to_local():
    cloud = FakeProvider("minimax", ("ok", "cloud"))
    local = FakeProvider("local-ollama", ("ok", "local"))
    eng = _engine(cloud, local)
    out = _call(eng, model="qwen2.5:3b-instruct-q4_K_M")
    assert out == "local"
    assert cloud.calls == []                      # cloud skipped for utility calls
    assert local.calls == ["qwen2.5:3b-instruct-q4_K_M"]


def test_build_chain_places_anthropic_before_local(monkeypatch):
    """When ANTHROPIC_API_KEY is set, Claude joins the chain after the other
    cloud providers and immediately before the local Ollama fallback."""
    from config import settings

    monkeypatch.setattr(settings, "minimax_api_key", "", raising=False)
    monkeypatch.setattr(settings, "groq_api_key", "", raising=False)
    monkeypatch.setattr(settings, "gemini_api_key", "g-key", raising=False)
    monkeypatch.setattr(settings, "anthropic_api_key", "a-key", raising=False)

    names = [p.name for p in FallbackLLM()._build()]
    assert names == ["gemini", "anthropic", "local-ollama"]


def test_all_providers_failing_raises():
    eng = _engine(FakeProvider("a", ("raise",)), FakeProvider("b", ("raise",)))
    try:
        _call(eng)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "All LLM providers failed" in str(exc)


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
