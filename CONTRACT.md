# Module Contract (implementation spec)

Source of truth for types: `core/models.py`. Config: `config.py` (`from config import settings`).
Qdrant layout: `core/schema.py`. Logging: `from core.logging import get_logger`.

**Conventions for all modules**
- Run from repo root; use absolute imports (`from config import settings`,
  `from core.models import Chunk`, `from ingest.embed import BGEM3Embedder`).
- Heavy models (torch / transformers / FlagEmbedding / surya) must be imported
  **lazily inside functions/methods or guarded**, so importing the module on a
  CPU-only box (and for `py_compile`) never fails. Top-of-file imports must be
  limited to stdlib + lightweight libs (pydantic, numpy, fitz, PIL).
- `from __future__ import annotations` at the top of every module.
- Use `get_logger(__name__)`; never `print` (except CLI scripts' user output).
- Add concise docstrings + type hints. Match the surrounding style.

---

## ingest/classify.py
```python
def classify_page(page, *, min_chars: int = settings.min_native_chars) -> PageKind
    # page: fitz.Page. NATIVE if extractable text length >= min_chars;
    # SCANNED if the page has image(s) but little/no text; EMPTY otherwise.

def classify_pdf(path: str | Path) -> list[PageKind]
    # Open with fitz, classify every page (1-based order preserved).
```

## ingest/extract.py
```python
def extract_native_text(page) -> str            # fitz page -> text (layout-aware)
def render_page_image(page, dpi: int = settings.ocr_dpi) -> "PIL.Image.Image"
def open_pdf(path: str | Path) -> "fitz.Document" # helper, caller closes
```

## ingest/normalize.py
```python
def normalize_arabic(text: str, *, strip_diacritics: bool = settings.strip_diacritics,
                     strip_tatweel: bool = settings.strip_tatweel) -> str
    # Unicode NFC; normalize alef forms (أإآٱ->ا), ya/alef-maqsura (ى->ي optional),
    # teh marbuta handling, remove tatweel ـ, collapse whitespace, optional
    # diacritics (harakat U+064B..U+0652) removal.
def normalize_text(text: str, lang: str | None = None) -> str
    # Apply arabic normalization when text contains Arabic; always tidy whitespace.
def detect_lang(text: str) -> str    # "ar" | "en" | "mixed" | "unknown" (langdetect, guarded)
```

## ingest/chunk.py
```python
def chunk_pages(pages: list[PageContent], book: BookMeta, *,
                target_tokens: int = settings.chunk_target_tokens,
                overlap_tokens: int = settings.chunk_overlap_tokens,
                min_tokens: int = settings.chunk_min_tokens) -> list[Chunk]
    # Concatenate page texts tracking page boundaries; split into ~target_tokens
    # chunks with overlap, sentence/paragraph-aware, never crossing the book.
    # token counting: approximate (e.g. whitespace/char heuristic, ~4 chars/token)
    # — document the heuristic. Set chunk_id = uuid5(f"{book.book_id}:{idx}"),
    # page_start/page_end from source pages, text = normalized text.
```

## ingest/embed.py
```python
class BGEM3Embedder:
    def __init__(self, model_name=settings.embedding_model, device=settings.embedding_device,
                 use_fp16=settings.embedding_use_fp16): ...   # lazy-load FlagEmbedding.BGEM3FlagModel
    def embed_documents(self, texts: list[str]) -> list[Embedding]
    def embed_query(self, text: str) -> Embedding
    # Use BGEM3FlagModel.encode(..., return_dense=True, return_sparse=True).
    # Convert lexical_weights dict -> SparseVector(indices, values).
```

## ingest/registry.py  (SQLite, stdlib sqlite3)
```python
class Registry:
    def __init__(self, db_path: Path = settings.registry_db): ...  # create table if missing
    @staticmethod
    def compute_hash(path: str | Path) -> str          # sha256 of file bytes
    def is_ingested(self, file_hash: str) -> bool       # status == 'completed'
    def get(self, file_hash: str) -> dict | None
    def mark_started(self, book: BookMeta) -> None
    def mark_completed(self, stats: IngestStats) -> None
    def mark_failed(self, book_id: str, error: str) -> None
    def list_books(self) -> list[dict]
```

## retrieval/search.py
```python
class QdrantStore:
    def __init__(self, url=settings.qdrant_url, api_key=settings.qdrant_api_key,
                 collection=schema.COLLECTION): ...
    def ensure_collection(self) -> None
        # Create collection with named dense vector (schema.DENSE_DIM, Cosine) +
        # sparse vector (schema.SPARSE_VECTOR); enable scalar quantization when
        # settings.qdrant_use_quantization; create payload indexes for book_id, lang.
    def upsert_chunks(self, chunks: list[Chunk], embeddings: list[Embedding],
                      batch_size: int = 128) -> int
        # point id = chunk_id (uuid string). Payload mirrors schema.PAYLOAD_FIELDS.
    def hybrid_search(self, query: Embedding, top_k: int = settings.search_top_k,
                      book_ids: list[str] | None = None) -> list[SearchResult]
        # Query dense + sparse; fuse with RRF (Qdrant Query API prefetch + FusionQuery,
        # or manual RRF with settings.rrf_k). Apply book_id filter when given.
        # Dense prefetch uses SearchParams(hnsw_ef=settings.qdrant_ef_search).
    def fetch_neighbors(self, book_id: str, chunk_indices: list[int]) -> list[SearchResult]
        # Exact-index scroll (book_id + level=passage + integer chunk_index) for
        # small-to-big context expansion.
    def delete_by_book(self, book_id: str) -> None
    def count(self) -> int
```

## retrieval/pipeline.py  (layers) + retrieval/expand.py
```python
def retrieve(question, *, embedder, store, reranker, route,
             book_ids=None, rerank_top_k=None) -> list[SearchResult]
    # API entrypoint: multi-query expansion (expand_queries -> variant embeddings)
    # + routed retrieval with RRF fusion across variant candidate lists.
def retrieve_for_route(question, embedding, store, reranker, route, *,
                       book_ids=None, rerank_top_k=None,
                       extra_embeddings=None) -> list[SearchResult]
    # Routed core (LOCAL passages / GLOBAL summaries+children). After rerank:
    # _apply_score_floor (settings.rerank_min_score) then expand_context
    # (stitch neighbours, overlap-aware, settings.context_window_chunks).
def rrf_fuse(result_lists, *, k=60) -> list[SearchResult]          # pure
def merge_overlapping(a: str, b: str, *, max_overlap=800) -> str   # pure
def expand_queries(question, *, max_variants=None) -> list[str]    # expand.py; never raises
```

## retrieval/rerank.py
```python
class Reranker:
    def __init__(self, model_name=settings.reranker_model, device=settings.reranker_device,
                 use_fp16=settings.reranker_use_fp16): ...   # lazy-load FlagEmbedding.FlagReranker
    def rerank(self, query: str, results: list[SearchResult],
               top_k: int = settings.rerank_top_k) -> list[SearchResult]
        # score (query, result.text) pairs; set rerank_score; return sorted top_k.
```

## llm/prompts.py
```python
SYSTEM_PROMPT: str   # grounded-answer instructions (bilingual AR/EN): answer ONLY
                     # from provided context, reply in the question's language,
                     # cite sources as [n], say "not found in the books" if unsupported.
def build_context_block(results: list[SearchResult]) -> str
    # Numbered sources: [n] "title" (author) p.start-end \n text
def build_user_prompt(question: str, results: list[SearchResult]) -> str
```

## llm/answer.py
```python
def answer_question(question: str, results: list[SearchResult], *,
                    model: str | None = None) -> Answer
    # anthropic.Anthropic(api_key=settings.anthropic_api_key).
    # System prompt sent with cache_control: {"type": "ephemeral"} (prompt caching).
    # model defaults to settings.answer_model. Parse [n] citations back to sources
    # -> Citation list. grounded=False when the model says it's not in the books.
def get_client() -> "anthropic.Anthropic"   # lazy singleton
```

## ingest/pipeline.py
```python
def ingest_book(path: str | Path, *, store: QdrantStore, embedder: BGEM3Embedder,
                registry: Registry, ocr=None, title: str | None = None,
                author: str | None = None) -> IngestStats
    # Full single-book flow: hash+dedup (skip if registry.is_ingested) ->
    # build BookMeta -> classify pages -> extract native / OCR scanned ->
    # normalize -> chunk -> embed -> upsert -> registry.mark_completed.
    # Catch per-page OCR errors into stats.failed_pages; never abort whole book.
def build_book_meta(path, file_hash, num_pages, title, author) -> BookMeta
```

## ingest/ocr.py
```python
class OCRBackend(Protocol/ABC):
    def ocr_image(self, image: "PIL.Image.Image") -> str
class QwenVLOCR(OCRBackend):  ...   # transformers Qwen2.5-VL, prompt to transcribe AR+EN
class SuryaOCR(OCRBackend):   ...   # surya recognition + detection
class TesseractOCR(OCRBackend): ... # pytesseract, lang=settings.ocr_lang
def get_ocr_backend(name: str = settings.ocr_backend) -> OCRBackend   # factory, lazy import
```

## ingest/worker.py  (RQ)
```python
def get_queue() -> "rq.Queue"                      # from settings.redis_url, settings.ingest_queue
def enqueue_book(path: str, **kw) -> str           # returns job id
def ingest_book_job(path: str, **kw) -> dict       # worker entrypoint: builds singletons, calls ingest_book, returns IngestStats.model_dump()
# `python -m ingest.worker` should start an RQ worker on the queue.
```

## api/main.py  (FastAPI)
```python
app = FastAPI(...)
GET  /health            -> {"status": "ok"}
GET  /status            -> {"books": int, "chunks": int}   # registry + qdrant counts
POST /ingest  (IngestRequest)  -> IngestResponse           # enqueue file/dir
POST /query   (QueryRequest)   -> QueryResponse            # embed->search->rerank->answer
POST /chat    (ChatRequest)    -> ChatResponse             # multi-turn, client-managed history
# Lazy-init singletons (embedder, reranker, store, conversations) on startup.
# Routers: api/routes_chat.py + api/routes_conversations.py + api/routes_books.py.
# GET / serves api/static/dist (React SPA) when built, else the legacy page;
# a catch-all GET /{path} serves the SPA shell for client routes (/dashboard).
```

## llm/errors.py
```python
class ProviderError(Exception)            # .provider, .reason, .message
# reason: "rate_limit" | "auth" | "timeout" | "unavailable" | "empty" | "error"
class AllProvidersFailedError(RuntimeError)
def classify_error(exc: Exception) -> str # duck-typed (status_code/name/message), no SDK imports
```

## llm/engine.py  (provider-aware additions)
```python
@dataclass GenResult: text: str; provider: str; model: str
def generate(system, messages, *, provider=None, model=None, ...) -> GenResult
# provider None/"auto": walk the fallback chain, report who answered.
# pinned id ("gemini"|"claude"|"local"): call only it; failure -> ProviderError (no fallback).
def stream(system, messages, *, provider=None, ...) -> Iterator[tuple[str, object]]
# yields ("provider", {provider, model}) once at first token, then ("delta", str)...
def list_providers() -> list[dict]        # [{id,label,model,available}] in chain order
def provider_label(provider_id: str) -> str
```

## core/conversations.py  (SQLite, stdlib sqlite3)
```python
class ConversationStore:                   # settings.conversations_db; WAL + FK cascade
    def create(title="", model="auto", book_ids=None) -> dict
    def list() -> list[dict]               # newest-updated first, with message_count
    def get(conv_id) -> dict | None        # includes ordered "messages"
    def append_message(conv_id, role, content, *, citations=None, model=None, grounded=None) -> dict
    def rename(conv_id, title) -> bool
    def delete(conv_id) -> bool
    def set_title_from_message(conv_id, text)  # word-boundary truncation, <= 60 chars
```

## api/routes_chat.py
```python
GET  /models       -> {"default": "auto", "providers": [...]}        # model picker source
POST /chat/stream  (ChatStreamRequest) -> SSE                        # server-side history
# events: meta {conversation_id, search_query, route, sources}
#         provider {provider, model}     (at first token)
#         delta {text}*                  done {citations, grounded, provider, model, conversation_id}
#         error {provider, reason, message, partial}   (terminal; pinned never falls back)
```

## api/routes_conversations.py
```python
GET/POST /conversations; GET/PATCH/DELETE /conversations/{id}        # CRUD over ConversationStore
```

## api/routes_books.py  (dashboard)
```python
POST   /upload            # multipart PDF -> uploads_dir, hash dedup, enqueue; 400/413 guards
GET    /jobs, /jobs/{id}  # RQ state + stage/current/total from job.meta ("not_found" is 200)
GET    /books             # registry rows + {total_books, total_chunks}
DELETE /books/{book_id}   # 409 mid-ingest; Qdrant delete_by_book + registry delete + unlink upload
```

## scripts/ingest_dir.py
```python
# CLI: python -m scripts.ingest_dir <dir> [--recursive] [--sync]
# Walk dir for *.pdf; by default enqueue via worker; --sync runs ingest_book inline.
```

## scripts/eval.py
```python
# CLI: python -m scripts.eval <questions.jsonl>
# Each line: {"question": ..., "expect_book_id"/"expect_page": optional}.
# Runs search+rerank, prints recall@k and (optionally) the generated answer.
```
