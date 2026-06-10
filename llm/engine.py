"""Pluggable text-generation engine — OFFLINE-FIRST.

The system answers with a **local model** by default; no cloud API is required.
Choose a backend with ``settings.llm_backend``:

* ``"transformers"`` (default): a local Hugging Face causal LM (e.g.
  Qwen2.5-Instruct) run in-process. Fully offline once the weights are cached.
* ``"openai"``: a local OpenAI-compatible server (vLLM / Ollama / LM Studio) at
  ``settings.openai_base_url`` — also offline (localhost), just a faster server.
* ``"anthropic"``: the Claude cloud API (opt-in; needs a key + network).

All heavy SDKs (torch/transformers, openai, anthropic) are imported lazily, so
this module stays importable on a CPU-only box. Public entrypoints:

* :func:`complete` — the original blocking call; signature and behavior are
  frozen (utility callers and the e2e fakes depend on it).
* :func:`generate` — like ``complete`` but returns a :class:`GenResult` naming
  the provider that actually answered, and supports *pinning* one provider
  (``provider="gemini"``), in which case failures raise
  :class:`~llm.errors.ProviderError` instead of falling through the chain.
* :func:`stream` — token streaming with the same auto/pinned semantics.
* :func:`list_providers` — the active chain as ``{id,label,model,available}``
  rows for the UI's model picker.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from config import settings
from core.logging import get_logger
from llm.errors import AllProvidersFailedError, ProviderError, classify_error

logger = get_logger(__name__)

_engine: Any = None
_lock = threading.Lock()


def active_model_name() -> str:
    """Human-readable name of the model the active backend will use.

    For the fallback chain this is the *preferred* (first available) provider's
    model; the actual answer may come from a later provider if earlier ones fail.
    """
    backend = settings.llm_backend
    if backend == "transformers":
        return settings.local_llm_model
    if backend == "openai":
        return settings.openai_model
    if backend == "anthropic":
        return settings.answer_model
    if backend == "fallback":
        if settings.minimax_api_key:
            return settings.minimax_model
        if settings.gemini_api_key:
            return settings.gemini_model
        if settings.groq_api_key:
            return settings.groq_model
        if settings.anthropic_api_key:
            return settings.answer_model
        return settings.openai_model
    return backend


# --- backends ---------------------------------------------------------------


class TransformersLLM:
    """Local Hugging Face causal LM, run in-process (fully offline)."""

    def __init__(self, provider_id: str = "local", label: str = "Local (Transformers)") -> None:
        self.provider_id = provider_id
        self.label = label
        self.model_name = settings.local_llm_model
        self._model: Any = None
        self._tok: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch  # lazy
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

        name = settings.local_llm_model
        logger.info("Loading local LLM %s on %s", name, settings.local_llm_device)
        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(settings.local_llm_dtype, "auto")
        on_cuda = settings.local_llm_device.startswith("cuda")

        self._tok = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForCausalLM.from_pretrained(
            name,
            torch_dtype=torch_dtype,
            device_map="auto" if on_cuda else None,
        )
        if not on_cuda:
            self._model = self._model.to(settings.local_llm_device)
        self._model.eval()

    def complete(self, system: str, messages: list[dict], *, model: str | None, max_tokens: int, temperature: float) -> str:
        import torch  # lazy

        self._load()
        chat = [{"role": "system", "content": system}] + list(messages)
        prompt = self._tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        inputs = self._tok([prompt], return_tensors="pt").to(self._model.device)

        do_sample = bool(temperature and temperature > 0)
        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": max_tokens,
            "do_sample": do_sample,
            "pad_token_id": self._tok.eos_token_id,
        }
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=0.9)

        with torch.inference_mode():
            output = self._model.generate(**inputs, **gen_kwargs)
        new_tokens = output[0][inputs.input_ids.shape[1]:]
        return self._tok.decode(new_tokens, skip_special_tokens=True).strip()

    def stream_complete(self, system: str, messages: list[dict], *, model: str | None, max_tokens: int, temperature: float) -> Iterator[str]:
        """In-process HF generation has no incremental API here — one-shot yield."""
        out = self.complete(system, messages, model=model, max_tokens=max_tokens, temperature=temperature)
        if out:
            yield out


class OpenAICompatibleLLM:
    """An OpenAI-compatible chat endpoint.

    Defaults to the local server (Ollama / vLLM / LM Studio) from ``settings``,
    but ``base_url``/``api_key``/``model`` can be supplied so the same class can
    drive a cloud provider (MiniMax, Gemini) inside :class:`FallbackLLM`.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        name: str = "openai",
        timeout: float | None = None,
        provider_id: str = "local",
        label: str = "Local (Ollama)",
    ) -> None:
        self._base_url = base_url or settings.openai_base_url
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_model
        self._timeout = timeout
        self.name = name                  # legacy chain name (tests match on it)
        self.provider_id = provider_id    # stable id exposed to the API/UI
        self.label = label                # human-readable picker label
        self.model_name = self._model
        self._client: Any = None

    def _client_or_load(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy

            logger.info("Using OpenAI-compatible endpoint %s (%s)", self._base_url, self.name)
            self._client = OpenAI(
                base_url=self._base_url, api_key=self._api_key, timeout=self._timeout
            )
        return self._client

    def complete(self, system: str, messages: list[dict], *, model: str | None, max_tokens: int, temperature: float) -> str:
        client = self._client_or_load()
        resp = client.chat.completions.create(
            model=model or self._model,
            messages=[{"role": "system", "content": system}] + list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream_complete(self, system: str, messages: list[dict], *, model: str | None, max_tokens: int, temperature: float) -> Iterator[str]:
        client = self._client_or_load()
        stream = client.chat.completions.create(
            model=model or self._model,
            messages=[{"role": "system", "content": system}] + list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
        )
        for chunk in stream:
            choices = getattr(chunk, "choices", None)
            if not choices:  # some providers send keep-alive/usage chunks
                continue
            text = getattr(choices[0].delta, "content", None)
            if text:
                yield text


class FallbackLLM:
    """Try cloud providers in order, then the local model (the offline guarantee).

    Chain: MiniMax -> Gemini -> local Ollama. A cloud provider is included only
    when its API key is set, and any error (bad key, network, incompatible
    response) is caught so the next provider is tried. The local model is always
    last, so answering never fully fails as long as Ollama is up.

    An explicit per-call ``model`` (used for cheap utility calls: condensation,
    routing, summarization) names a LOCAL model, so it is sent straight to the
    local provider and never burns cloud quota.
    """

    def __init__(self) -> None:
        self._providers: list[OpenAICompatibleLLM] | None = None

    def _build(self) -> list["OpenAICompatibleLLM"]:
        if self._providers is not None:
            return self._providers
        providers: list[OpenAICompatibleLLM] = []
        if settings.minimax_api_key:
            providers.append(OpenAICompatibleLLM(
                base_url=settings.minimax_base_url, api_key=settings.minimax_api_key,
                model=settings.minimax_model, name="minimax", timeout=settings.cloud_llm_timeout,
                provider_id="minimax", label="MiniMax",
            ))
        if settings.gemini_api_key:
            providers.append(OpenAICompatibleLLM(
                base_url=settings.gemini_base_url, api_key=settings.gemini_api_key,
                model=settings.gemini_model, name="gemini", timeout=settings.cloud_llm_timeout,
                provider_id="gemini", label="Gemini",
            ))
        if settings.groq_api_key:
            providers.append(OpenAICompatibleLLM(
                base_url=settings.groq_base_url, api_key=settings.groq_api_key,
                model=settings.groq_model, name="groq", timeout=settings.cloud_llm_timeout,
                provider_id="groq", label="Groq",
            ))
        # Claude (cloud) sits after the other cloud providers and before the local
        # model — e.g. Gemini primary, Claude as fallback if Gemini fails.
        if settings.anthropic_api_key:
            providers.append(AnthropicLLM(
                name="anthropic", timeout=settings.cloud_llm_timeout,
                provider_id="claude", label="Claude",
            ))
        providers.append(OpenAICompatibleLLM(
            base_url=settings.openai_base_url, api_key=settings.openai_api_key,
            model=settings.openai_model, name="local-ollama", timeout=settings.local_llm_timeout,
            provider_id="local", label="Local (Ollama)",
        ))
        self._providers = providers
        logger.info("LLM fallback chain: %s", " -> ".join(p.name for p in providers))
        return providers

    def complete(self, system: str, messages: list[dict], *, model: str | None, max_tokens: int, temperature: float) -> str:
        providers = self._build()

        # An explicit model override names a LOCAL model — go straight local.
        if model is not None:
            return providers[-1].complete(
                system, messages, model=model, max_tokens=max_tokens, temperature=temperature
            )

        errors: list[str] = []
        for provider in providers:
            try:
                out = provider.complete(
                    system, messages, model=None, max_tokens=max_tokens, temperature=temperature
                )
                if out and out.strip():
                    if errors:
                        logger.info(
                            "Answered via %s after %d provider(s) failed", provider.name, len(errors)
                        )
                    return out
                errors.append(f"{provider.name}: empty response")
            except Exception as exc:  # noqa: BLE001 - try the next provider
                logger.warning("LLM provider %s failed (%s); trying next", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
        raise AllProvidersFailedError("All LLM providers failed: " + "; ".join(errors))


class AnthropicLLM:
    """Claude cloud API. Used as the configured backend (``llm_backend=anthropic``)
    or as a link in the :class:`FallbackLLM` chain. Requires ANTHROPIC_API_KEY."""

    def __init__(
        self,
        name: str = "anthropic",
        timeout: float | None = None,
        provider_id: str = "claude",
        label: str = "Claude",
    ) -> None:
        self.name = name
        self.provider_id = provider_id
        self.label = label
        self.model_name = settings.answer_model
        self._timeout = timeout
        self._client: Any = None

    def _client_or_load(self) -> Any:
        if self._client is None:
            import anthropic  # lazy

            kwargs: dict[str, Any] = {"api_key": settings.anthropic_api_key}
            if self._timeout is not None:
                kwargs["timeout"] = self._timeout
            self._client = anthropic.Anthropic(**kwargs)
        return self._client

    def complete(self, system: str, messages: list[dict], *, model: str | None, max_tokens: int, temperature: float) -> str:
        client = self._client_or_load()
        resp = client.messages.create(
            model=model or settings.answer_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=list(messages),
        )
        parts: list[str] = []
        for block in getattr(resp, "content", None) or []:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            if btype == "text":
                txt = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
                if txt:
                    parts.append(txt)
        return "".join(parts).strip()

    def stream_complete(self, system: str, messages: list[dict], *, model: str | None, max_tokens: int, temperature: float) -> Iterator[str]:
        client = self._client_or_load()
        # Same cached system block as complete() so streaming hits the cache too.
        with client.messages.stream(
            model=model or settings.answer_model,
            max_tokens=max_tokens,
            temperature=temperature,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=list(messages),
        ) as stream:
            for text in stream.text_stream:
                if text:
                    yield text


_BACKENDS = {
    "transformers": TransformersLLM,
    "openai": OpenAICompatibleLLM,
    "anthropic": AnthropicLLM,
    "fallback": FallbackLLM,
}


def get_engine() -> Any:
    """Construct (once) and return the configured backend instance."""
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                cls = _BACKENDS.get(settings.llm_backend)
                if cls is None:
                    raise ValueError(
                        f"Unknown llm_backend {settings.llm_backend!r}; "
                        f"choose one of: {', '.join(_BACKENDS)}"
                    )
                logger.info("LLM backend: %s (model=%s)", settings.llm_backend, active_model_name())
                _engine = cls()
    return _engine


def complete(
    system: str,
    messages: list[dict],
    *,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    """Generate a completion from ``system`` + chat ``messages`` via the backend.

    ``messages`` is a list of ``{"role": "user"|"assistant", "content": str}``.
    Returns the assistant's text. Defaults pull from ``settings`` when unset.
    """
    eng = get_engine()
    return eng.complete(
        system,
        messages,
        model=model,
        max_tokens=max_tokens or settings.answer_max_tokens,
        temperature=settings.answer_temperature if temperature is None else temperature,
    )


# --- Provider-aware API (auto fallback / pinned provider / streaming) ---------


@dataclass
class GenResult:
    """A completion plus the identity of the provider/model that produced it."""

    text: str
    provider: str   # provider_id, e.g. "gemini" | "claude" | "local"
    model: str      # the model that actually answered


def _chain() -> list[Any]:
    """Provider instances in fallback order for the active backend.

    The fallback backend exposes its full chain; any single backend is a
    one-element chain, so auto/pinned logic works uniformly.
    """
    eng = get_engine()
    if isinstance(eng, FallbackLLM):
        return eng._build()
    return [eng]


def _provider_model(provider: Any) -> str:
    """The provider's configured model name (best effort for duck-typed fakes)."""
    return getattr(provider, "model_name", None) or getattr(provider, "_model", None) or ""


def provider_label(provider_id: str) -> str:
    """Human-readable label for a provider id (the id itself when unknown)."""
    for p in _chain():
        if getattr(p, "provider_id", None) == provider_id:
            return getattr(p, "label", provider_id)
    return provider_id


# Cached local-server probe: (monotonic timestamp, reachable?). The probe is
# only consulted for the "local" provider so the picker can grey it out when
# Ollama is down; it must never raise or block the request path noticeably.
_LOCAL_PROBE_TTL = 60.0
_local_probe: tuple[float, bool] | None = None


def _local_available() -> bool:
    global _local_probe
    now = time.monotonic()
    if _local_probe is not None and now - _local_probe[0] < _LOCAL_PROBE_TTL:
        return _local_probe[1]
    ok = False
    try:
        import urllib.request  # stdlib; lazy to keep import time trivial

        url = settings.openai_base_url.rstrip("/") + "/models"
        with urllib.request.urlopen(url, timeout=1.5):
            ok = True
    except Exception:  # noqa: BLE001 - the probe must never raise
        ok = False
    _local_probe = (now, ok)
    return ok


def list_providers() -> list[dict]:
    """The active chain as ``{id, label, model, available}`` rows, in order."""
    rows: list[dict] = []
    for p in _chain():
        pid = getattr(p, "provider_id", "local")
        rows.append({
            "id": pid,
            "label": getattr(p, "label", pid),
            "model": _provider_model(p),
            "available": _local_available() if pid == "local" else True,
        })
    return rows


def _resolve_pinned(chain: list[Any], provider: str) -> Any:
    """The chain entry whose provider_id matches, else ProviderError."""
    for p in chain:
        if getattr(p, "provider_id", None) == provider:
            return p
    known = ", ".join(getattr(p, "provider_id", "?") for p in chain)
    raise ProviderError(provider, "error", f"unknown provider {provider!r}; available: {known}")


def generate(
    system: str,
    messages: list[dict],
    *,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> GenResult:
    """Like :func:`complete`, but provider-aware.

    ``provider=None``/``"auto"`` walks the chain exactly like the fallback
    backend (an explicit ``model=`` still means "go straight to the local /
    last provider" — the utility-model rule) and reports who answered. Any
    other ``provider`` PINS that provider: it alone is called, and any failure
    raises :class:`ProviderError` — no silent fallback.
    """
    max_tokens = max_tokens or settings.answer_max_tokens
    temperature = settings.answer_temperature if temperature is None else temperature
    chain = _chain()

    if provider is not None and provider != "auto":
        target = _resolve_pinned(chain, provider)
        try:
            out = target.complete(system, messages, model=model, max_tokens=max_tokens, temperature=temperature)
        except Exception as exc:  # noqa: BLE001 - pinned: classify, never fall back
            raise ProviderError(provider, classify_error(exc), str(exc)) from exc
        if not out or not out.strip():
            raise ProviderError(provider, "empty", f"{provider} returned an empty response")
        return GenResult(text=out, provider=provider, model=model or _provider_model(target))

    # Auto: an explicit model override names a LOCAL model — go straight to the
    # last (local) provider, mirroring FallbackLLM.complete.
    if model is not None:
        last = chain[-1]
        out = last.complete(system, messages, model=model, max_tokens=max_tokens, temperature=temperature)
        return GenResult(text=out, provider=getattr(last, "provider_id", "local"), model=model)

    errors: list[str] = []
    for p in chain:
        try:
            out = p.complete(system, messages, model=None, max_tokens=max_tokens, temperature=temperature)
            if out and out.strip():
                if errors:
                    logger.info("Answered via %s after %d provider(s) failed", p.name, len(errors))
                return GenResult(text=out, provider=getattr(p, "provider_id", p.name), model=_provider_model(p))
            errors.append(f"{p.name}: empty response")
        except Exception as exc:  # noqa: BLE001 - try the next provider
            logger.warning("LLM provider %s failed (%s); trying next", p.name, exc)
            errors.append(f"{p.name}: {exc}")
    raise AllProvidersFailedError("All LLM providers failed: " + "; ".join(errors))


def stream(
    system: str,
    messages: list[dict],
    *,
    provider: str | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> Iterator[tuple[str, object]]:
    """Stream a completion as ``("provider", {...})`` then ``("delta", text)...``.

    The ``provider`` event is emitted exactly once, only after the FIRST text
    chunk has been pulled successfully — in auto mode a provider that dies
    before its first chunk is skipped (next in chain), so callers never see a
    provider announced that then produced nothing. Failures AFTER the first
    chunk raise :class:`ProviderError` (callers already hold partial output).
    Pinned mode never falls back: any failure raises :class:`ProviderError`.
    """
    max_tokens = max_tokens or settings.answer_max_tokens
    temperature = settings.answer_temperature if temperature is None else temperature
    chain = _chain()

    pinned = provider is not None and provider != "auto"
    if pinned:
        candidates = [_resolve_pinned(chain, provider)]
    elif model is not None:
        candidates = [chain[-1]]  # explicit model = local/last, as in generate()
    else:
        candidates = chain

    errors: list[str] = []
    for p in candidates:
        pid = getattr(p, "provider_id", p.name)
        it = p.stream_complete(system, messages, model=model, max_tokens=max_tokens, temperature=temperature)
        try:
            first = next(chunk for chunk in it if chunk)
        except StopIteration:  # stream ended before any text
            if pinned:
                raise ProviderError(pid, "empty", f"{pid} returned an empty response") from None
            logger.warning("LLM provider %s streamed nothing; trying next", p.name)
            errors.append(f"{p.name}: empty response")
            continue
        except Exception as exc:  # noqa: BLE001 - failed before the first chunk
            if pinned:
                raise ProviderError(pid, classify_error(exc), str(exc)) from exc
            logger.warning("LLM provider %s failed (%s); trying next", p.name, exc)
            errors.append(f"{p.name}: {exc}")
            continue

        if errors:
            logger.info("Streaming via %s after %d provider(s) failed", p.name, len(errors))
        yield ("provider", {"provider": pid, "model": model or _provider_model(p)})
        yield ("delta", first)
        # Past the first chunk there is no falling back — partial output has
        # already reached the caller, so surface failures as ProviderError.
        while True:
            try:
                chunk = next(it)
            except StopIteration:
                return
            except Exception as exc:  # noqa: BLE001
                raise ProviderError(pid, classify_error(exc), str(exc)) from exc
            if chunk:
                yield ("delta", chunk)

    raise AllProvidersFailedError("All LLM providers failed: " + "; ".join(errors))
