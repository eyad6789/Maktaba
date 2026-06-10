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
    # Backend: "openai" (local Ollama / vLLM / LM Studio server — CPU-friendly,
    # the default) | "fallback" (try cloud providers MiniMax -> Gemini, then the
    # local model — best quality online, still works offline) | "transformers"
    # (local in-process HF, needs a GPU) | "anthropic" (Claude cloud, opt-in).
    # The default targets a quantized GGUF model served by Ollama, the only
    # practical way to run a 7B answer model on CPU. For a GPU box, set
    # LLM_BACKEND=transformers. To prefer cloud quality, set LLM_BACKEND=fallback
    # and provide MINIMAX_API_KEY / GEMINI_API_KEY below.
    llm_backend: str = "openai"
    answer_max_tokens: int = 1024
    answer_temperature: float = 0.2                  # low = grounded & professional

    # Adaptive generation for whole-book / thematic (GLOBAL) answers: more room
    # to synthesize and a touch more freedom than tight factual (LOCAL) lookups.
    answer_max_tokens_global: int = 1536
    answer_temperature_global: float = 0.35
    # Query routing. The heuristic is free (no model call); the LLM tie-breaker
    # is opt-in and only runs when the heuristic is unsure.
    router_use_llm: bool = False

    # transformers backend (local model, no API). Qwen2.5-Instruct is strong on
    # Arabic + English; scale the size to your GPU (7B / 14B / 32B / 72B). Used
    # only when llm_backend="transformers" (a GPU box).
    local_llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    local_llm_device: str = "cuda"                   # "cpu" for small models in dev
    local_llm_dtype: str = "auto"                    # auto | float16 | bfloat16 | float32

    # openai-compatible local server backend (Ollama / vLLM / LM Studio). Default
    # points at Ollama. `openai_model` answers user questions; `utility_model` is
    # a smaller/faster model for high-volume cheap calls (summarization at ingest,
    # query condensation, routing). Both are quantized GGUF tags for CPU.
    openai_base_url: str = "http://localhost:11434/v1"   # Ollama (vLLM: :8000/v1)
    openai_api_key: str = "not-needed"               # local servers ignore this
    openai_model: str = "qwen2.5:7b-instruct-q4_K_M"
    utility_model: str = "qwen2.5:3b-instruct-q4_K_M"

    # Cloud fallback providers (used only when llm_backend="fallback"). The chain
    # tries MiniMax, then Gemini, then the local Ollama model above — each cloud
    # provider is SKIPPED unless its API key is set, and any failure falls through
    # to the next, so the system always degrades gracefully to fully-offline local.
    # Both providers expose an OpenAI-compatible /chat/completions endpoint.
    minimax_api_key: str = Field(default="", alias="MINIMAX_API_KEY")
    minimax_base_url: str = "https://api.minimax.io/v1"
    minimax_model: str = "MiniMax-M2"
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    # flash-lite returns clean answers within the token cap; plain 2.5-flash is a
    # "thinking" model that can spend the budget on reasoning and truncate.
    gemini_model: str = "gemini-2.5-flash-lite"
    # Groq — a genuinely free, very fast OpenAI-compatible provider. Easiest free
    # path to professional answers: get a key at https://console.groq.com/keys
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    groq_base_url: str = "https://api.groq.com/openai/v1"
    groq_model: str = "llama-3.3-70b-versatile"
    # Per-provider request timeouts (seconds). Cloud is short so a dead provider
    # fails fast and the chain moves on; local is long because CPU is slow.
    cloud_llm_timeout: float = 60.0
    local_llm_timeout: float = 600.0

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

    # --- Embeddings (BGE-M3) ---
    # CPU-first defaults (fp16 is a no-op / unstable on CPU). On a GPU box set
    # EMBEDDING_DEVICE=cuda and EMBEDDING_USE_FP16=true.
    embedding_model: str = "BAAI/bge-m3"
    embedding_dim: int = 1024                        # BGE-M3 dense dimension
    embedding_batch_size: int = 12
    embedding_device: str = "cpu"                    # "cuda" on a GPU box
    embedding_max_length: int = 8192
    embedding_use_fp16: bool = False                 # True only with CUDA

    # --- Reranker (bge-reranker-v2-m3) ---
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"                     # "cuda" on a GPU box
    reranker_use_fp16: bool = False                  # True only with CUDA
    rerank_top_k: int = 8                            # results kept after reranking

    # --- Retrieval ---
    search_top_k: int = 50                           # candidates from Qdrant pre-rerank
    rrf_k: int = 60                                  # Reciprocal Rank Fusion constant
    dense_weight: float = 1.0
    sparse_weight: float = 1.0

    # GLOBAL (whole-book / thematic) route: retrieve chapter+book summary nodes,
    # keep the best few, then drill into their child passages. Tuned small so the
    # answer prompt stays within a CPU model's context.
    global_summary_k: int = 20                       # summary candidates from Qdrant
    global_summary_keep: int = 4                     # summaries kept after rerank
    global_child_keep: int = 6                       # child passages kept after rerank

    # --- OCR (scanned pages) ---
    # CPU default is tesseract (qwen-vl/surya need a GPU to be practical). On a
    # GPU box set OCR_BACKEND=qwen for best Arabic OCR quality.
    ocr_backend: str = "tesseract"                   # qwen | surya | tesseract
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

    # --- Comprehension layer (hierarchical chapter + book summaries) ---
    # When enabled, ingestion detects each book's structure (PDF table of
    # contents) and builds chapter-level and book-level summary nodes with the
    # small `summary_model`, stored alongside raw chunks. This is what lets the
    # bot answer whole-book / thematic questions as if it understood the book.
    # OFF by default (building summaries is many LLM calls — a one-time, opt-in
    # re-ingest). See README for the runbook.
    enable_comprehension: bool = False
    enable_heading_detection: bool = False           # heuristic fallback when no TOC
    summary_model: str = "qwen2.5:3b-instruct-q4_K_M"  # small/fast; = utility_model
    summary_map_batch_tokens: int = 2500             # map step batches up to this
    summary_max_tokens: int = 768                    # length cap per summary node
    summary_min_section_tokens: int = 300            # skip summarizing tiny sections

    # --- Arabic normalization ---
    strip_diacritics: bool = False                   # keep for classical/religious texts
    strip_tatweel: bool = True

    # --- Paths (relative to repo root) ---
    data_dir: Path = Path("data")
    books_dir: Path = Path("data/books")
    registry_db: Path = Path("data/registry/registry.db")
    log_level: str = "INFO"

    # --- Conversations (server-side chat history) ---
    conversations_db: Path = Path("data/registry/conversations.db")

    # --- Browser uploads (dashboard) ---
    uploads_dir: Path = Path("data/books/uploads")
    max_upload_mb: int = 200


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
