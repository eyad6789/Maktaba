"""The OpenAI-compatible backend routes a per-call `model` (Phase 1).

This is what lets a single backend serve a small `utility_model` for cheap calls
(summaries, condensation, routing) and a larger model for answers. We stub the
HTTP client so no server/network/`openai` package is needed — only the dispatch
logic in `OpenAICompatibleLLM.complete` is exercised.
"""

from __future__ import annotations

from config import settings
from llm.engine import OpenAICompatibleLLM


class _FakeResp:
    class _Choice:
        class _Msg:
            content = "ok"

        message = _Msg()

    choices = [_Choice()]


class _FakeClient:
    """Records the kwargs passed to chat.completions.create."""

    def __init__(self, sink: dict) -> None:
        self._sink = sink
        self.chat = self  # so client.chat.completions.create resolves
        self.completions = self

    def create(self, **kwargs):  # noqa: ANN003
        self._sink.update(kwargs)
        return _FakeResp()


def _backend_with_sink() -> tuple[OpenAICompatibleLLM, dict]:
    sink: dict = {}
    backend = OpenAICompatibleLLM()
    backend._client = _FakeClient(sink)  # bypass lazy openai import
    return backend, sink


def test_explicit_model_is_forwarded():
    backend, sink = _backend_with_sink()
    out = backend.complete(
        "sys", [{"role": "user", "content": "hi"}],
        model="my-utility-model", max_tokens=8, temperature=0.0,
    )
    assert out == "ok"
    assert sink["model"] == "my-utility-model"


def test_missing_model_falls_back_to_settings():
    backend, sink = _backend_with_sink()
    backend.complete(
        "sys", [{"role": "user", "content": "hi"}],
        model=None, max_tokens=8, temperature=0.0,
    )
    assert sink["model"] == settings.openai_model


def test_system_and_messages_are_passed_through():
    backend, sink = _backend_with_sink()
    backend.complete(
        "SYSTEM", [{"role": "user", "content": "Q"}],
        model="m", max_tokens=8, temperature=0.2,
    )
    msgs = sink["messages"]
    assert msgs[0] == {"role": "system", "content": "SYSTEM"}
    assert msgs[-1] == {"role": "user", "content": "Q"}
    assert sink["temperature"] == 0.2


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
