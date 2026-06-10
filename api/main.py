"""FastAPI service for the Arabic/English book RAG system.

Exposes the query and ingestion surface of the system:

* ``GET  /health`` — liveness probe.
* ``GET  /status`` — book/chunk counts (registry + Qdrant).
* ``POST /ingest`` — enqueue a PDF file or directory for async ingestion.
* ``POST /query``  — embed → hybrid search → rerank → grounded answer.
* ``POST /chat``   — multi-turn chat (client-managed history, non-streaming).

Mounted routers extend this surface:

* :mod:`api.routes_chat`          — ``GET /models``, ``POST /chat/stream`` (SSE).
* :mod:`api.routes_conversations` — server-side conversation CRUD.
* :mod:`api.routes_books`         — dashboard: upload / jobs / books / delete.

The React SPA (built into ``api/static/dist`` by ``web/``) is served at ``/``;
when no build is present the legacy single-file UI at ``api/static/index.html``
is served instead, so the API works without Node.

Heavy singletons (the BGE-M3 embedder, the cross-encoder reranker and the
Qdrant store) are constructed once during the FastAPI lifespan and reused
across requests; the answer LLM client and the ingestion queue are imported
lazily inside request handlers so module import (and ``py_compile``) stays
light and side-effect free on a CPU-only box.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import TYPE_CHECKING, AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from config import settings
from core.logging import get_logger
from core.models import (
    ChatRequest,
    ChatResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from ingest.embed import BGEM3Embedder
    from ingest.registry import Registry
    from retrieval.rerank import Reranker
    from retrieval.search import QdrantStore

logger = get_logger(__name__)

# PDF discovery for directory ingestion.
_PDF_SUFFIX = ".pdf"

# Static web chatbot UI shipped alongside the API. The React SPA build (from
# web/) lands in static/dist; without it the legacy index.html is served.
_STATIC_DIR = Path(__file__).resolve().parent / "static"
_DIST_DIR = _STATIC_DIR / "dist"


# -- lifespan / singleton wiring ---------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Construct shared singletons on startup; tear nothing down on shutdown.

    The Qdrant store's collection is ensured to exist here so the first query
    does not race collection creation. The embedder and reranker are cheap to
    construct (their models load lazily on first use), so building them up-front
    keeps the heavy model load off the request path's critical section while
    still avoiding import-time side effects.
    """
    from core.conversations import ConversationStore
    from ingest.embed import BGEM3Embedder
    from ingest.registry import Registry
    from retrieval.rerank import Reranker
    from retrieval.search import QdrantStore

    logger.info("Initializing RAG service singletons")

    store = QdrantStore()
    store.ensure_collection()

    app.state.store = store
    app.state.embedder = BGEM3Embedder()
    app.state.reranker = Reranker()
    app.state.registry = Registry()
    app.state.conversations = ConversationStore()

    logger.info("RAG service ready (collection=%s)", store.collection)
    try:
        yield
    finally:
        logger.info("Shutting down RAG service")


app = FastAPI(
    title="Arabic/English Book RAG",
    description=(
        "Hybrid (dense + sparse) retrieval over ingested PDF books with "
        "cross-encoder reranking and grounded, cited answers from Claude."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# -- state accessors ----------------------------------------------------------


def _get_store(request: Request) -> "QdrantStore":
    store = getattr(request.app.state, "store", None)
    if store is None:  # pragma: no cover - defensive; lifespan always sets it
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return store


def _get_embedder(request: Request) -> "BGEM3Embedder":
    embedder = getattr(request.app.state, "embedder", None)
    if embedder is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Embedder not initialized")
    return embedder


def _get_reranker(request: Request) -> "Reranker":
    reranker = getattr(request.app.state, "reranker", None)
    if reranker is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Reranker not initialized")
    return reranker


def _get_registry(request: Request) -> "Registry":
    registry = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return registry


# -- endpoints ----------------------------------------------------------------


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the web UI: the built React SPA, or the legacy page without it."""
    spa_index = _DIST_DIR / "index.html"
    if spa_index.is_file():
        return FileResponse(spa_index)
    return FileResponse(_STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, str]:
    """Liveness probe: always returns ``{"status": "ok"}`` once serving."""
    return {"status": "ok"}


@app.get("/status")
def status(request: Request) -> dict[str, int]:
    """Report counts of ingested books (registry) and chunks (Qdrant)."""
    registry = _get_registry(request)
    store = _get_store(request)
    try:
        books = len(registry.list_books())
    except Exception as exc:  # noqa: BLE001 - report degraded rather than 500
        logger.warning("status: registry.list_books failed: %s", exc)
        books = 0
    chunks = store.count()
    return {"books": books, "chunks": chunks}


def _discover_pdfs(path: Path, recursive: bool) -> list[Path]:
    """Return the PDF files implied by ``path``.

    A file path yields itself (when it is a ``.pdf``); a directory is walked
    for ``*.pdf`` (recursively when ``recursive`` is set, otherwise one level).
    Results are sorted for deterministic ordering.
    """
    if path.is_file():
        return [path] if path.suffix.lower() == _PDF_SUFFIX else []
    if path.is_dir():
        pattern = "**/*.pdf" if recursive else "*.pdf"
        return sorted(p for p in path.glob(pattern) if p.is_file())
    return []


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: Request, req: IngestRequest) -> IngestResponse:
    """Enqueue a PDF file or a directory of PDFs for asynchronous ingestion.

    Already-completed files (matched by content hash in the registry) are
    reported under ``skipped`` and not re-enqueued; the rest are pushed onto
    the RQ ingest queue and their job ids returned under ``enqueued``.
    """
    # Lazy imports keep Redis/RQ off the import path of this module.
    from ingest.worker import enqueue_book

    registry = _get_registry(request)

    root = Path(req.path).expanduser()
    if not root.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {req.path}")

    pdfs = _discover_pdfs(root, req.recursive)
    if not pdfs:
        raise HTTPException(
            status_code=400,
            detail=f"No PDF files found at: {req.path}",
        )

    enqueued: list[str] = []
    skipped: list[str] = []
    for pdf in pdfs:
        abs_path = str(pdf.resolve())
        try:
            file_hash = registry.compute_hash(abs_path)
            if registry.is_ingested(file_hash):
                logger.info("Skipping already-ingested file: %s", abs_path)
                skipped.append(abs_path)
                continue
        except Exception as exc:  # noqa: BLE001 - hashing failures shouldn't 500
            logger.warning("Could not hash %s (%s); enqueuing anyway", abs_path, exc)

        try:
            job_id = enqueue_book(abs_path)
        except Exception as exc:  # noqa: BLE001 - queue may be unreachable
            logger.error("Failed to enqueue %s: %s", abs_path, exc)
            raise HTTPException(
                status_code=503,
                detail=f"Could not enqueue ingestion job: {exc}",
            ) from exc
        enqueued.append(job_id)

    return IngestResponse(
        enqueued=enqueued,
        skipped=skipped,
        count=len(enqueued),
    )


@app.post("/query", response_model=QueryResponse)
def query(request: Request, req: QueryRequest) -> QueryResponse:
    """Answer a question grounded in the ingested books.

    Pipeline: embed the question (dense + sparse) → hybrid search Qdrant for
    candidates → cross-encoder rerank → generate a cited, grounded answer with
    Claude. ``top_k``, ``book_ids`` and ``model`` from the request override the
    corresponding defaults.
    """
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question must not be empty")

    # Lazy import: keeps heavy/LLM deps out of module import.
    from llm.answer import answer_question
    from llm.errors import ProviderError
    from retrieval.pipeline import retrieve_for_route
    from retrieval.route import classify_route, coerce_route

    embedder = _get_embedder(request)
    store = _get_store(request)
    reranker = _get_reranker(request)

    # Route the question (whole-book/thematic vs factual), then retrieve over the
    # appropriate index levels. An explicit req.route overrides the classifier.
    route = coerce_route(req.route) or classify_route(question, req.book_ids)
    embedding = embedder.embed_query(question)
    rerank_top_k = req.top_k if req.top_k and req.top_k > 0 else settings.rerank_top_k
    reranked = retrieve_for_route(
        question, embedding, store, reranker, route,
        book_ids=req.book_ids, rerank_top_k=rerank_top_k,
    )

    # A pinned provider does NOT fall back: its failure surfaces as 429/502 so
    # the client can tell the user to switch models or use Auto.
    try:
        answer = answer_question(
            question, reranked, model=req.model, route=route, provider=req.provider
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=429 if exc.reason == "rate_limit" else 502,
            detail={"provider": exc.provider, "reason": exc.reason, "message": exc.message},
        ) from exc

    return QueryResponse(
        answer=answer.answer,
        citations=answer.citations,
        sources=answer.sources,
        model=answer.model,
        grounded=answer.grounded,
        provider=answer.provider,
    )


@app.post("/chat", response_model=ChatResponse)
def chat(request: Request, req: ChatRequest) -> ChatResponse:
    """Multi-turn chat grounded in the ingested books.

    Condenses the conversation into a standalone retrieval query (so follow-ups
    work), runs hybrid search + rerank, then answers the latest message with the
    prior turns in context and inline ``[n]`` citations.
    """
    if not req.messages or req.messages[-1].role != "user" or not req.messages[-1].content.strip():
        raise HTTPException(status_code=400, detail="last message must be a non-empty user turn")

    # Lazy import: keeps heavy/LLM deps out of module import.
    from llm.chat import chat_answer, condense_query
    from llm.errors import ProviderError
    from retrieval.pipeline import retrieve_for_route
    from retrieval.route import classify_route, coerce_route

    embedder = _get_embedder(request)
    store = _get_store(request)
    reranker = _get_reranker(request)

    messages = [m.model_dump() for m in req.messages]
    # Condensation is a cheap rewrite — always use the small utility model, even
    # when req.model overrides the (larger) model used for the actual answer.
    search_query = (
        condense_query(messages, model=settings.utility_model)
        if req.condense
        else req.messages[-1].content.strip()
    )

    # Route on the (condensed) standalone query, then retrieve over levels.
    route = coerce_route(req.route) or classify_route(search_query, req.book_ids)
    embedding = embedder.embed_query(search_query)
    rerank_top_k = req.top_k if req.top_k and req.top_k > 0 else settings.rerank_top_k
    reranked = retrieve_for_route(
        search_query, embedding, store, reranker, route,
        book_ids=req.book_ids, rerank_top_k=rerank_top_k,
    )

    # A pinned provider does NOT fall back: its failure surfaces as 429/502 so
    # the client can tell the user to switch models or use Auto.
    try:
        answer = chat_answer(
            messages, reranked, model=req.model, route=route, provider=req.provider
        )
    except ProviderError as exc:
        raise HTTPException(
            status_code=429 if exc.reason == "rate_limit" else 502,
            detail={"provider": exc.provider, "reason": exc.reason, "message": exc.message},
        ) from exc

    return ChatResponse(
        answer=answer.answer,
        citations=answer.citations,
        sources=answer.sources,
        model=answer.model,
        grounded=answer.grounded,
        search_query=search_query,
        provider=answer.provider,
    )


# -- routers + SPA serving ------------------------------------------------------
# Registration order matters: API routers first, the SPA catch-all last, so
# client-side routes like /dashboard resolve to the SPA without shadowing any
# API path.

from api.routes_books import router as books_router  # noqa: E402
from api.routes_chat import router as chat_router  # noqa: E402
from api.routes_conversations import router as conversations_router  # noqa: E402

app.include_router(chat_router)
app.include_router(conversations_router)
app.include_router(books_router)

if (_DIST_DIR / "assets").is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_DIST_DIR / "assets"), name="assets")


@app.get("/{path:path}", include_in_schema=False)
def spa_fallback(path: str) -> FileResponse:
    """Serve the SPA shell for client-side routes (e.g. /dashboard, /c/<id>).

    Real files under the build dir (favicons etc.) are served directly; paths
    that look like files but don't exist 404 instead of returning HTML.
    """
    spa_index = _DIST_DIR / "index.html"
    if not spa_index.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    candidate = (_DIST_DIR / path).resolve()
    if candidate.is_file() and candidate.is_relative_to(_DIST_DIR.resolve()):
        return FileResponse(candidate)
    if "." in path.rsplit("/", maxsplit=1)[-1]:
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(spa_index)


def main() -> None:
    """Run the API with uvicorn (``python -m api.main``)."""
    import uvicorn

    host = os.environ.get("API_HOST", "0.0.0.0")
    port = int(os.environ.get("API_PORT", "8000"))
    logger.info("Starting API server on %s:%d", host, port)
    uvicorn.run(app, host=host, port=port, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
