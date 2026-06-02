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
    """Human-readable name of the model the active backend will use."""
    backend = settings.llm_backend
    if backend == "transformers":
        return settings.local_llm_model
    if backend == "openai":
        return settings.openai_model
    if backend == "anthropic":
        return settings.answer_model
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
    """A local OpenAI-compatible server (vLLM / Ollama / LM Studio)."""

    def __init__(self) -> None:
        self._client: Any = None

    def _client_or_load(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy

            logger.info("Using OpenAI-compatible endpoint %s", settings.openai_base_url)
            self._client = OpenAI(base_url=settings.openai_base_url, api_key=settings.openai_api_key)
        return self._client

    def complete(self, system: str, messages: list[dict], *, model: str | None, max_tokens: int, temperature: float) -> str:
        client = self._client_or_load()
        resp = client.chat.completions.create(
            model=model or settings.openai_model,
            messages=[{"role": "system", "content": system}] + list(messages),
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()


class AnthropicLLM:
    """Claude cloud API (opt-in). Requires ANTHROPIC_API_KEY + network."""

    def __init__(self) -> None:
        self._client: Any = None

    def _client_or_load(self) -> Any:
        if self._client is None:
            import anthropic  # lazy

            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
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
