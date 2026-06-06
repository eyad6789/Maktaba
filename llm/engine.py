"""Pluggable text-generation engine — OFFLINE-FIRST.

The system answers with a **local model** by default; no cloud API is required.
Choose a backend with ``settings.llm_backend``:

* ``"transformers"`` (default): a local Hugging Face causal LM (e.g.
  Qwen2.5-Instruct) run in-process. Fully offline once the weights are cached.
* ``"openai"``: a local OpenAI-compatible server (vLLM / Ollama / LM Studio) at
  ``settings.openai_base_url`` — also offline (localhost), just a faster server.
* ``"anthropic"``: the Claude cloud API (opt-in; needs a key + network).

All heavy SDKs (torch/transformers, openai, anthropic) are imported lazily, so
this module stays importable on a CPU-only box. The single public entrypoint is
:func:`complete`.
"""

from __future__ import annotations

import threading
from typing import Any

from config import settings
from core.logging import get_logger

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

    def __init__(self) -> None:
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
    ) -> None:
        self._base_url = base_url or settings.openai_base_url
        self._api_key = api_key or settings.openai_api_key
        self._model = model or settings.openai_model
        self._timeout = timeout
        self.name = name
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
            ))
        if settings.gemini_api_key:
            providers.append(OpenAICompatibleLLM(
                base_url=settings.gemini_base_url, api_key=settings.gemini_api_key,
                model=settings.gemini_model, name="gemini", timeout=settings.cloud_llm_timeout,
            ))
        if settings.groq_api_key:
            providers.append(OpenAICompatibleLLM(
                base_url=settings.groq_base_url, api_key=settings.groq_api_key,
                model=settings.groq_model, name="groq", timeout=settings.cloud_llm_timeout,
            ))
        # Claude (cloud) sits after the other cloud providers and before the local
        # model — e.g. Gemini primary, Claude as fallback if Gemini fails.
        if settings.anthropic_api_key:
            providers.append(AnthropicLLM(name="anthropic", timeout=settings.cloud_llm_timeout))
        providers.append(OpenAICompatibleLLM(
            base_url=settings.openai_base_url, api_key=settings.openai_api_key,
            model=settings.openai_model, name="local-ollama", timeout=settings.local_llm_timeout,
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
        raise RuntimeError("All LLM providers failed: " + "; ".join(errors))


class AnthropicLLM:
    """Claude cloud API. Used as the configured backend (``llm_backend=anthropic``)
    or as a link in the :class:`FallbackLLM` chain. Requires ANTHROPIC_API_KEY."""

    def __init__(self, name: str = "anthropic", timeout: float | None = None) -> None:
        self.name = name
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
