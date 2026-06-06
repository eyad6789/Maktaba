"""CPU-first default settings (Phase 1).

These guard the offline/CPU defaults so a regression that re-points the system at
a GPU-only or cloud backend is caught immediately. They read the live `settings`
singleton, which on CI (no `.env`) reflects the in-code defaults.
"""

from __future__ import annotations

from config import settings


def test_llm_backend_is_local_capable():
    # "openai" = a LOCAL Ollama/vLLM server; "fallback" = cloud chain that ends at
    # the local model. Both are CPU-friendly and never require a GPU or a key.
    assert settings.llm_backend in ("openai", "fallback")


def test_openai_endpoint_points_at_local_ollama():
    assert settings.openai_base_url.startswith("http://localhost")
    assert "/v1" in settings.openai_base_url


def test_dual_model_present_and_quantized():
    # A utility model is configured for cheap high-volume calls. It may equal the
    # answer model on a small single-model box (a valid deployment), so we only
    # require it to be set and to be a quantized GGUF tag runnable on CPU.
    assert settings.utility_model
    assert "q4" in settings.openai_model.lower()
    assert "q4" in settings.utility_model.lower()


def test_embeddings_and_reranker_default_to_cpu():
    assert settings.embedding_device == "cpu"
    assert settings.reranker_device == "cpu"
    # fp16 is unstable / a no-op on CPU — must be off by default.
    assert settings.embedding_use_fp16 is False
    assert settings.reranker_use_fp16 is False


def test_ocr_backend_defaults_to_cpu_friendly_tesseract():
    assert settings.ocr_backend == "tesseract"


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
