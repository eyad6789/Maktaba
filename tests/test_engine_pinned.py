"""Provider-aware engine API: generate() / stream() / list_providers().

Auto mode keeps the fallback walk (and the explicit-model = straight-to-local
rule); PINNED mode calls exactly one provider and raises ProviderError on any
failure — never falling back. Providers are faked and the chain is
monkeypatched in, so no settings, network, or SDK packages are needed.
"""

from __future__ import annotations

import pytest

import llm.engine as engine
from llm.engine import GenResult
from llm.errors import AllProvidersFailedError, ProviderError


class _RateLimited(Exception):
    """A 429-shaped SDK error (recognized via the status_code attribute)."""

    status_code = 429


class FakeProvider:
    def __init__(self, provider_id, behavior=("ok", "text"), *, model="fake-model", chunks=None):
        # behavior: ("ok", text) | ("raise", exc) | ("empty",)
        # chunks (streaming): ("chunks", [..]) | ("raise_before", exc)
        #                     | ("raise_after", first_chunk, exc)
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
        mode = self.chunks[0]
        if mode == "raise_before":
            raise self.chunks[1]
        if mode == "raise_after":
            yield self.chunks[1]
            raise self.chunks[2]
        yield from self.chunks[1]


def _patch_chain(monkeypatch, providers):
    monkeypatch.setattr(engine, "_chain", lambda: list(providers))
    return providers


def _gen(**kwargs):
    return engine.generate("sys", [{"role": "user", "content": "q"}],
                           max_tokens=64, temperature=0.2, **kwargs)


def _stream(**kwargs):
    return engine.stream("sys", [{"role": "user", "content": "q"}],
                         max_tokens=64, temperature=0.2, **kwargs)


# --- generate(): pinned mode --------------------------------------------------


def test_pinned_success_returns_genresult_for_that_provider(monkeypatch):
    gemini = FakeProvider("gemini", ("ok", "from gemini"), model="gemini-2.5")
    local = FakeProvider("local", ("ok", "from local"))
    _patch_chain(monkeypatch, [gemini, local])

    res = _gen(provider="gemini")
    assert res == GenResult(text="from gemini", provider="gemini", model="gemini-2.5")
    assert local.calls == []  # only the pinned provider is touched


def test_pinned_rate_limit_raises_and_never_falls_back(monkeypatch):
    gemini = FakeProvider("gemini", ("raise", _RateLimited("too many requests")))
    local = FakeProvider("local", ("ok", "from local"))
    _patch_chain(monkeypatch, [gemini, local])

    with pytest.raises(ProviderError) as exc_info:
        _gen(provider="gemini")
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.reason == "rate_limit"
    assert local.calls == []  # the point of pinning: NO silent fallback


def test_pinned_empty_output_raises_empty(monkeypatch):
    claude = FakeProvider("claude", ("empty",))
    _patch_chain(monkeypatch, [claude])

    with pytest.raises(ProviderError) as exc_info:
        _gen(provider="claude")
    assert exc_info.value.reason == "empty"


def test_pinned_unknown_provider_rejected(monkeypatch):
    _patch_chain(monkeypatch, [FakeProvider("gemini")])

    with pytest.raises(ProviderError) as exc_info:
        _gen(provider="does-not-exist")
    assert exc_info.value.provider == "does-not-exist"
    assert exc_info.value.reason == "error"


# --- generate(): auto mode ----------------------------------------------------


def test_auto_reports_the_answering_provider(monkeypatch):
    gemini = FakeProvider("gemini", ("raise", RuntimeError("down")))
    claude = FakeProvider("claude", ("ok", "from claude"), model="claude-sonnet")
    _patch_chain(monkeypatch, [gemini, claude])

    res = _gen()  # provider=None == auto
    assert res.text == "from claude"
    assert res.provider == "claude"
    assert res.model == "claude-sonnet"
    assert gemini.calls and claude.calls  # walked in order


def test_auto_explicit_model_goes_straight_to_last(monkeypatch):
    gemini = FakeProvider("gemini", ("ok", "cloud"))
    local = FakeProvider("local", ("ok", "local answer"))
    _patch_chain(monkeypatch, [gemini, local])

    res = _gen(provider="auto", model="qwen2.5:3b-instruct-q4_K_M")
    assert res == GenResult(text="local answer", provider="local", model="qwen2.5:3b-instruct-q4_K_M")
    assert gemini.calls == []  # utility-model rule: cloud never burned
    assert local.calls == ["qwen2.5:3b-instruct-q4_K_M"]


def test_auto_all_failing_raises_all_providers_failed(monkeypatch):
    _patch_chain(monkeypatch, [
        FakeProvider("gemini", ("raise", RuntimeError("down"))),
        FakeProvider("local", ("empty",)),
    ])
    with pytest.raises(AllProvidersFailedError, match="^All LLM providers failed"):
        _gen()


# --- stream() -------------------------------------------------------------


def test_stream_auto_falls_back_before_first_chunk(monkeypatch):
    dead = FakeProvider("gemini", chunks=("raise_before", _RateLimited("429")))
    live = FakeProvider("claude", model="claude-sonnet", chunks=("chunks", ["Hello ", "world"]))
    _patch_chain(monkeypatch, [dead, live])

    events = list(_stream())
    assert events[0] == ("provider", {"provider": "claude", "model": "claude-sonnet"})
    assert [d for kind, d in events[1:] if kind == "delta"] == ["Hello ", "world"]
    assert dead.stream_calls and live.stream_calls  # both attempted, in order


def test_stream_failure_after_first_chunk_raises_provider_error(monkeypatch):
    flaky = FakeProvider("gemini", model="gemini-2.5",
                         chunks=("raise_after", "partial ", _RateLimited("quota")))
    backup = FakeProvider("claude", chunks=("chunks", ["never used"]))
    _patch_chain(monkeypatch, [flaky, backup])

    received = []
    with pytest.raises(ProviderError) as exc_info:
        for event in _stream():
            received.append(event)
    # The provider event and the partial delta arrived before the failure...
    assert received == [
        ("provider", {"provider": "gemini", "model": "gemini-2.5"}),
        ("delta", "partial "),
    ]
    # ...and post-first-chunk errors never fall back (partial output is out).
    assert exc_info.value.provider == "gemini"
    assert exc_info.value.reason == "rate_limit"
    assert backup.stream_calls == []


def test_stream_pinned_failure_raises_instead_of_falling_back(monkeypatch):
    dead = FakeProvider("gemini", chunks=("raise_before", _RateLimited("429")))
    backup = FakeProvider("local", chunks=("chunks", ["should not stream"]))
    _patch_chain(monkeypatch, [dead, backup])

    with pytest.raises(ProviderError) as exc_info:
        list(_stream(provider="gemini"))
    assert exc_info.value.reason == "rate_limit"
    assert backup.stream_calls == []


# --- list_providers() -------------------------------------------------------


def test_list_providers_in_chain_order(monkeypatch):
    _patch_chain(monkeypatch, [
        FakeProvider("gemini", model="gemini-2.5"),
        FakeProvider("claude", model="claude-sonnet"),
        FakeProvider("local", model="qwen2.5:7b"),
    ])
    monkeypatch.setattr(engine, "_local_available", lambda: False)

    rows = engine.list_providers()
    assert [r["id"] for r in rows] == ["gemini", "claude", "local"]
    assert [r["model"] for r in rows] == ["gemini-2.5", "claude-sonnet", "qwen2.5:7b"]
    assert [r["available"] for r in rows] == [True, True, False]  # local probed
    assert rows[0]["label"] == "Gemini"


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
