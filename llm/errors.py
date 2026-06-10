"""Provider error taxonomy for the LLM engine.

Two exception types plus a duck-typed classifier:

* :class:`ProviderError` — a single *pinned* provider failed (or streamed and
  then died). Carries the provider id and a coarse ``reason`` so the API layer
  can map it to a status code and the UI can say "Gemini hit its rate limit —
  switch model or use Auto" instead of silently falling back.
* :class:`AllProvidersFailedError` — the auto fallback chain was exhausted.
  Subclasses ``RuntimeError`` and keeps the historical
  ``"All LLM providers failed: ..."`` message, so existing callers (and tests)
  that catch ``RuntimeError`` and match on that prefix keep working.

:func:`classify_error` is deliberately duck-typed — it inspects ``status_code``
attributes, type names, and message text rather than importing the openai /
anthropic SDKs, so this module is importable (and testable in CI) on a box with
none of the cloud SDKs installed.
"""

from __future__ import annotations

# The closed set of coarse failure reasons a ProviderError may carry.
REASONS: tuple[str, ...] = (
    "rate_limit",   # 429 / quota exhausted — retry later or switch provider
    "auth",         # 401/403 — bad or missing API key
    "timeout",      # request timed out
    "unavailable",  # 5xx / overloaded — provider-side outage
    "empty",        # provider answered with blank output
    "error",        # anything else
)

# Message substrings (lowercased) that signal a rate-limit / quota failure even
# when the exception carries no status code (e.g. Gemini's RESOURCE_EXHAUSTED).
_RATE_LIMIT_TOKENS: tuple[str, ...] = ("rate limit", "rate_limit", "quota", "resource_exhausted", "429")


class ProviderError(Exception):
    """A specific LLM provider failed; no fallback was (or may be) attempted."""

    def __init__(self, provider: str, reason: str, message: str) -> None:
        self.provider = provider
        self.reason = reason
        self.message = message
        super().__init__(f"[{provider}] {reason}: {message}")


class AllProvidersFailedError(RuntimeError):
    """Every provider in the auto fallback chain failed or returned nothing.

    Constructed with the same ``"All LLM providers failed: ..."`` message text
    the engine has always raised — do not change that prefix.
    """


def classify_error(exc: BaseException) -> str:
    """Map an arbitrary provider exception to one of :data:`REASONS`.

    Duck-typed on purpose: openai/anthropic/httpx errors all expose a
    ``status_code`` attribute and conventional type names ("RateLimitError",
    "APITimeoutError"), so no SDK import is needed to classify them.
    """
    status = getattr(exc, "status_code", None)
    if status == 429:
        return "rate_limit"
    if status in (401, 403):
        return "auth"
    if status in (500, 502, 503, 529):
        return "unavailable"

    if "RateLimit" in type(exc).__name__:
        return "rate_limit"
    text = str(exc).lower()
    if any(token in text for token in _RATE_LIMIT_TOKENS):
        return "rate_limit"
    if isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__:
        return "timeout"
    return "error"


__all__ = ["ProviderError", "AllProvidersFailedError", "classify_error", "REASONS"]
