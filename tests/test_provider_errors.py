"""Duck-typed provider error classification (llm/errors.py).

`classify_error` must recognize SDK-style exceptions WITHOUT importing any SDK:
it goes by `status_code` attributes, type names, and message text. Everything
here uses hand-rolled fake exceptions, so the test runs on the light CI deps.
"""

from __future__ import annotations

from llm.errors import AllProvidersFailedError, ProviderError, classify_error


class _StatusError(Exception):
    """Shaped like openai/anthropic APIStatusError: carries `status_code`."""

    def __init__(self, status_code: int, msg: str = "boom") -> None:
        super().__init__(msg)
        self.status_code = status_code


class RateLimitError(Exception):
    """Recognized by type NAME alone, like the SDKs' RateLimitError classes."""


class ReadTimeout(Exception):
    """Recognized by 'Timeout' in the type name (httpx.ReadTimeout shape)."""


def test_status_429_is_rate_limit():
    assert classify_error(_StatusError(429)) == "rate_limit"


def test_status_401_and_403_are_auth():
    assert classify_error(_StatusError(401)) == "auth"
    assert classify_error(_StatusError(403)) == "auth"


def test_status_5xx_and_529_are_unavailable():
    for code in (500, 502, 503, 529):
        assert classify_error(_StatusError(code)) == "unavailable"


def test_rate_limit_by_type_name():
    assert classify_error(RateLimitError("nope")) == "rate_limit"


def test_rate_limit_by_message_text():
    # Gemini says RESOURCE_EXHAUSTED; others say quota / rate limit / 429.
    for msg in (
        "Quota exceeded for model",
        "You hit your rate limit, slow down",
        "rate_limit_error: too many tokens",
        "RESOURCE_EXHAUSTED: per-minute cap",
        "HTTP 429 from upstream",
    ):
        assert classify_error(Exception(msg)) == "rate_limit"


def test_timeout_by_isinstance_and_type_name():
    assert classify_error(TimeoutError("slow")) == "timeout"
    assert classify_error(ReadTimeout("read timed out")) == "timeout"


def test_generic_falls_back_to_error():
    assert classify_error(Exception("connection refused")) == "error"
    assert classify_error(ValueError("bad json")) == "error"


def test_provider_error_carries_provider_reason_message():
    err = ProviderError("gemini", "rate_limit", "too many requests")
    assert err.provider == "gemini"
    assert err.reason == "rate_limit"
    assert err.message == "too many requests"
    assert "gemini" in str(err) and "too many requests" in str(err)


def test_all_providers_failed_is_runtime_error_with_prefix():
    err = AllProvidersFailedError("All LLM providers failed: gemini: down")
    assert isinstance(err, RuntimeError)
    assert str(err).startswith("All LLM providers failed")


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
