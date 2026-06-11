"""Provider cooldown for the AUTO fallback chain (llm/engine.py).

A provider that fails with a rate-limit-classified error goes on cooldown for
``settings.provider_cooldown_seconds`` and subsequent AUTO chain calls skip it;
per-DAY quota errors escalate the window to >= 30 minutes. PINNED calls bypass
cooldowns entirely (neither consulted nor recorded), and a fully-cooled chain
is still attempted in order — cooldown is an optimization, not a breaker.

The engine clock (``engine._now``) is injected so no test sleeps; providers are
faked and the chain monkeypatched in, so no network or SDK packages are needed.
"""

from __future__ import annotations

import pytest

import llm.engine as engine
from llm.engine import FallbackLLM
from llm.errors import AllProvidersFailedError, ProviderError


class _RateLimited(Exception):
    """A 429-shaped SDK error (recognized via the status_code attribute)."""

    status_code = 429


# The real incident: Gemini free tier exhausted its PER-DAY request quota.
_DAILY_QUOTA_MSG = (
    "Error code: 429 - quota exceeded for metric "
    "GenerateRequestsPerDayPerProjectPerModel-FreeTier. Please retry in 3600s."
)


class FakeProvider:
    def __init__(self, provider_id, behavior=("ok", "text"), *, model="fake-model", chunks=None):
        # behavior: ("ok", text) | ("raise", exc) | ("empty",)
        self.provider_id = provider_id
        self.name = provider_id
        self.label = provider_id.title()
        self.model_name = model
        self.behavior = behavior
        self.chunks = chunks or ("chunks", ["hi"])
        self.calls: list = []
        self.stream_calls: list = []

    def complete(self, system, messages, *, model, max_tokens, temperature):
        self.calls.append(model)
        kind = self.behavior[0]
        if kind == "raise":
            raise self.behavior[1]
        if kind == "empty":
            return ""
        return self.behavior[1]

    def stream_complete(self, system, messages, *, model, max_tokens, temperature):
        self.stream_calls.append(model)
        yield from self.chunks[1]


class FakeClock:
    def __init__(self, start: float = 1000.0) -> None:
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


@pytest.fixture
def clock(monkeypatch):
    """Inject a controllable monotonic clock and start with no cooldowns."""
    clk = FakeClock()
    monkeypatch.setattr(engine, "_now", clk)
    engine.reset_cooldowns()
    yield clk
    engine.reset_cooldowns()


def _patch_chain(monkeypatch, providers):
    monkeypatch.setattr(engine, "_chain", lambda: list(providers))
    return providers


def _gen(**kwargs):
    return engine.generate("sys", [{"role": "user", "content": "q"}],
                           max_tokens=64, temperature=0.2, **kwargs)


# --- skip + recovery ----------------------------------------------------------


def test_rate_limited_provider_skipped_within_window(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", _RateLimited("429 too many requests")))
    local = FakeProvider("local", ("ok", "from local"))
    _patch_chain(monkeypatch, [gemini, local])

    assert _gen().text == "from local"    # first call: gemini tried, fails, cools
    assert len(gemini.calls) == 1

    clock.advance(60)                     # still inside the 120s default window
    assert _gen().text == "from local"
    assert len(gemini.calls) == 1         # skipped — never re-called
    assert len(local.calls) == 2


def test_provider_retried_after_window_elapses(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", _RateLimited("429")))
    local = FakeProvider("local", ("ok", "from local"))
    _patch_chain(monkeypatch, [gemini, local])

    _gen()
    clock.advance(121)                    # past provider_cooldown_seconds=120
    _gen()
    assert len(gemini.calls) == 2         # tried again once the window elapsed


def test_non_rate_limit_failure_does_not_cool_down(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", RuntimeError("connection refused")))
    local = FakeProvider("local", ("ok", "from local"))
    _patch_chain(monkeypatch, [gemini, local])

    _gen()
    _gen()                                # no clock advance: still tried each call
    assert len(gemini.calls) == 2


def test_daily_quota_escalates_cooldown(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", Exception(_DAILY_QUOTA_MSG)))
    local = FakeProvider("local", ("ok", "from local"))
    _patch_chain(monkeypatch, [gemini, local])

    _gen()
    clock.advance(300)                    # well past the 120s default...
    _gen()
    assert len(gemini.calls) == 1         # ...but the per-day window (1800s) holds

    clock.advance(1500)                   # 1800s elapsed in total
    _gen()
    assert len(gemini.calls) == 2         # escalated window over — tried again


# --- pinned calls bypass cooldowns entirely ------------------------------------


def test_pinned_call_ignores_cooldown(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", _RateLimited("429")))
    local = FakeProvider("local", ("ok", "from local"))
    _patch_chain(monkeypatch, [gemini, local])

    _gen()                                # auto call puts gemini on cooldown
    assert len(gemini.calls) == 1

    gemini.behavior = ("ok", "gemini recovered")
    res = _gen(provider="gemini")         # pinned: called even while cooled down
    assert res.text == "gemini recovered"
    assert len(gemini.calls) == 2


def test_pinned_rate_limit_does_not_record_cooldown(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", _RateLimited("429")))
    local = FakeProvider("local", ("ok", "from local"))
    _patch_chain(monkeypatch, [gemini, local])

    with pytest.raises(ProviderError) as exc_info:
        _gen(provider="gemini")           # pinned failure: raises, no fallback...
    assert exc_info.value.reason == "rate_limit"

    _gen()                                # ...and the next AUTO call still tries gemini
    assert len(gemini.calls) == 2


# --- graceful degradation -------------------------------------------------------


def test_all_providers_cooled_down_still_attempts_chain(monkeypatch, clock):
    a = FakeProvider("gemini", ("raise", _RateLimited("429")))
    b = FakeProvider("local", ("raise", _RateLimited("429")))
    _patch_chain(monkeypatch, [a, b])

    with pytest.raises(AllProvidersFailedError):
        _gen()                            # both fail and go on cooldown
    with pytest.raises(AllProvidersFailedError, match="^All LLM providers failed"):
        _gen()                            # fully-cooled chain walked anyway, in order
    assert len(a.calls) == 2 and len(b.calls) == 2


def test_last_provider_not_skipped_when_alone(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", _RateLimited("429")))
    local = FakeProvider("local", ("raise", _RateLimited("429")))
    _patch_chain(monkeypatch, [gemini, local])

    with pytest.raises(AllProvidersFailedError):
        _gen()                            # both cooled now
    local.behavior = ("ok", "local back up")
    assert _gen().text == "local back up"  # cooled-but-only chain still answers


# --- the other AUTO chain walks share the same cooldowns ------------------------


def test_fallbackllm_complete_respects_cooldown(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", _RateLimited("429")))
    local = FakeProvider("local-ollama", ("ok", "offline answer"))
    eng = FallbackLLM()
    eng._providers = [gemini, local]      # bypass _build (no settings/network)

    kwargs = dict(model=None, max_tokens=64, temperature=0.2)
    assert eng.complete("sys", [{"role": "user", "content": "q"}], **kwargs) == "offline answer"
    assert eng.complete("sys", [{"role": "user", "content": "q"}], **kwargs) == "offline answer"
    assert len(gemini.calls) == 1         # second call skipped the cooled provider


def test_stream_auto_skips_cooled_provider(monkeypatch, clock):
    gemini = FakeProvider("gemini", ("raise", _RateLimited("429")), chunks=("chunks", ["never"]))
    local = FakeProvider("local", ("ok", "x"), model="qwen2.5:7b", chunks=("chunks", ["Hello"]))
    _patch_chain(monkeypatch, [gemini, local])

    _gen()                                # cool gemini down via generate()
    events = list(engine.stream("sys", [{"role": "user", "content": "q"}],
                                max_tokens=64, temperature=0.2))
    assert events[0] == ("provider", {"provider": "local", "model": "qwen2.5:7b"})
    assert gemini.stream_calls == []      # skipped without being called


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
