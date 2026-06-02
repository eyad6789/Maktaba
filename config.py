"""Central configuration for the Arabic/English book RAG system.

All settings are overridable via environment variables or a `.env` file at the
repo root. Import `settings` anywhere: `from config import settings`.

Run all entrypoints from the repo root so absolute imports
(`from config import ...`, `from core.models import ...`) resolve.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    # --- Answer LLM (OFFLINE by default) ---
    # Backend: "transformers" (local in-process, offline) | "openai"
    # (local vLLM/Ollama/LM Studio server) | "anthropic" (cloud, opt-in).
    llm_backend: str = "transformers"
    answer_max_tokens: int = 1024
    answer_temperature: float = 0.2                  # low = grounded & professional

    # transformers backend (local model, no API). Qwen2.5-Instruct is strong on
    # Arabic + English; scale the size to your GPU (7B / 14B / 32B / 72B).
    local_llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    local_llm_device: str = "cuda"                   # "cpu" for small models in dev
    local_llm_dtype: str = "auto"                    # auto | float16 | bfloat16 | float32

    # openai-compatible local server backend (vLLM / Ollama / LM Studio).
    openai_base_url: str = "http://localhost:8000/v1"
    openai_api_key: str = "not-needed"               # local servers ignore this
    openai_model: str = "Qwen/Qwen2.5-7B-Instruct"

    # anthropic backend (opt-in only; unused unless llm_backend="anthropic").
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    answer_model: str = "claude-sonnet-4-6"

    # --- Qdrant (vector DB, self-hosted) ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "books"
    qdrant_use_quantization: bool = True             # scalar quantization to cut RAM

    # --- Redis / ingestion queue ---
    redis_url: str = "redis://localhost:6379/0"
    ingest_queue: str = "ingest"

    # --- Embeddings (BGE-M3, local GPU) ---
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024                        # BGE-M3 dense dimension
    embedding_batch_size: int = 12
    embedding_device: str = "cuda"                   # "cpu" for local dev
    embedding_max_length: int = 8192
    embedding_use_fp16: bool = True

    # --- Reranker (bge-reranker-v2-m3, local GPU) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cuda"
    reranker_use_fp16: bool = True
    rerank_top_k: int = 8                            # results kept after reranking

    # --- Retrieval ---
    search_top_k: int = 50                           # candidates from Qdrant pre-rerank
    rrf_k: int = 60                                  # Reciprocal Rank Fusion constant
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # --- OCR (scanned pages, local GPU) ---
    ocr_backend: str = "qwen"                        # qwen | surya | tesseract
    ocr_dpi: int = 300                               # render DPI for scanned pages
    ocr_lang: str = "ara+eng"                        # tesseract language string
    qwen_ocr_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    qwen_ocr_device: str = "cuda"
    qwen_ocr_max_new_tokens: int = 2048
    ocr_max_workers: int = 1                         # parallel OCR workers (GPU-bound)

    # --- Page classification (native text vs scanned) ---
    min_native_chars: int = 100                      # chars/page below which we OCR

    # --- Chunking ---
    chunk_target_tokens: int = 600
    chunk_overlap_tokens: int = 90
    chunk_min_tokens: int = 50

    # --- Arabic normalization ---
    strip_diacritics: bool = False                   # keep for classical/religious texts
    strip_tatweel: bool = True

    # --- Paths (relative to repo root) ---
    data_dir: Path = Path("data")
    books_dir: Path = Path("data/books")
    registry_db: Path = Path("data/registry/registry.db")
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
