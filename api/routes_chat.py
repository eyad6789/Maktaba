"""Streaming chat + model listing.

* ``GET  /models``      — the model picker's source of truth: Auto plus every
  provider in the active fallback chain, with availability.
* ``POST /chat/stream`` — one chat turn as Server-Sent Events. The server owns
  conversation history (see :mod:`core.conversations`); the client sends only
  the new message. Event order: ``meta`` (sources, before generation) →
  ``provider`` (who is answering, known at first token) → ``delta``\\* →
  ``done`` | ``error``.

A pinned provider that fails does NOT fall back — the failure is surfaced as
an ``error`` event so the UI can tell the user to switch models or use Auto.
Heavy LLM/retrieval imports stay inside handlers (CPU-box import hygiene,
matching :mod:`api.main`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from config import settings
from core.logging import get_logger
from core.models import ChatStreamRequest
from core.sse import format_sse

if TYPE_CHECKING:  # pragma: no cover - typing only
    from core.conversations import ConversationStore
    from ingest.embed import BGEM3Embedder
    from retrieval.rerank import Reranker
    from retrieval.search import QdrantStore

logger = get_logger(__name__)

router = APIRouter()


# -- state accessors (mirror api.main; local copies avoid a circular import) ---


def _state(request: Request, name: str, detail: str):
    value = getattr(request.app.state, name, None)
    if value is None:  # pragma: no cover - defensive; lifespan always sets these
        raise HTTPException(status_code=503, detail=detail)
    return value


def _get_conversations(request: Request) -> "ConversationStore":
    return _state(request, "conversations", "Conversation store not initialized")


def _get_embedder(request: Request) -> "BGEM3Embedder":
    return _state(request, "embedder", "Embedder not initialized")


def _get_store(request: Request) -> "QdrantStore":
    return _state(request, "store", "Vector store not initialized")


def _get_reranker(request: Request) -> "Reranker":
    return _state(request, "reranker", "Reranker not initialized")


# -- endpoints ------------------------------------------------------------------


@router.get("/models")
def models() -> dict:
    """Providers for the model picker: Auto first, then the fallback chain."""
    from llm import engine

    providers = engine.list_providers()
    return {
        "default": "auto",
        "providers": [
            {
                "id": "auto",
                "label": "Auto",
                "model": None,
                "available": True,
                "chain": [p["id"] for p in providers],
            },
            *providers,
        ],
    }


def _human_error(provider: str | None, reason: str, message: str) -> str:
    """A user-facing message for a provider failure (shown in the chat UI)."""
    from llm import engine

    label = engine.provider_label(provider) if provider else "The model"
    if reason == "rate_limit":
        return f"{label} hit its rate limit — switch model or use Auto."
    if reason == "auth":
        return f"{label} rejected the configured API key."
    if reason == "timeout":
        return f"{label} timed out before finishing."
    if reason == "empty":
        return f"{label} returned an empty answer — try again or switch model."
    if reason == "unavailable":
        return f"{label} is currently unavailable — switch model or use Auto."
    return f"{label} failed: {message}"


@router.post("/chat/stream")
def chat_stream(request: Request, req: ChatStreamRequest) -> StreamingResponse:
    """Answer one chat turn over SSE, persisting both turns server-side."""
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=400, detail="message must not be empty")

    conversations = _get_conversations(request)
    embedder = _get_embedder(request)
    store = _get_store(request)
    reranker = _get_reranker(request)

    # Resolve the conversation before streaming starts so client errors get a
    # real HTTP status instead of a 200 stream that immediately errors.
    if req.conversation_id:
        conv = conversations.get(req.conversation_id)
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found")
    else:
        conv = conversations.create(model=req.provider or "auto", book_ids=req.book_ids)
    conv_id = conv["id"]

    # Per-turn overrides fall back to the conversation's stored defaults.
    provider = req.provider or conv.get("model") or "auto"
    book_ids = req.book_ids if req.book_ids is not None else conv.get("book_ids")

    history = [
        {"role": m["role"], "content": m["content"]}
        for m in conv.get("messages") or []
    ]
    messages = history + [{"role": "user", "content": message}]

    # Persist the user turn immediately — it survives even if generation fails,
    # so the user can retry (possibly with another model).
    conversations.append_message(conv_id, "user", message)
    conversations.set_title_from_message(conv_id, message)

    def gen() -> Iterator[str]:
        # Lazy imports: keep heavy LLM/retrieval deps off module import.
        from llm import engine
        from llm.answer import extract_citations
        from llm.chat import build_chat_messages, condense_query
        from llm.errors import AllProvidersFailedError, ProviderError
        from retrieval.pipeline import retrieve
        from retrieval.route import Route, classify_route, coerce_route

        try:
            search_query = (
                condense_query(messages, model=settings.utility_model)
                if req.condense
                else message
            )
            route = coerce_route(req.route) or classify_route(search_query, book_ids)
            rerank_top_k = (
                req.top_k if req.top_k and req.top_k > 0 else settings.rerank_top_k
            )
            reranked = retrieve(
                search_query, embedder=embedder, store=store, reranker=reranker,
                route=route, book_ids=book_ids, rerank_top_k=rerank_top_k,
            )

            # Sources are shown to the user before generation begins — on a
            # slow chain this is the difference between "working" and "stuck".
            yield format_sse("meta", {
                "conversation_id": conv_id,
                "search_query": search_query,
                "route": route.value,
                "provider_requested": provider,
                "sources": [s.model_dump() for s in reranked],
            })

            if not reranked:
                text = "This information was not found in the books."
                conversations.append_message(
                    conv_id, "assistant", text, citations=[], grounded=False,
                )
                yield format_sse("delta", {"text": text})
                yield format_sse("done", {
                    "citations": [], "grounded": False, "provider": None,
                    "model": None, "conversation_id": conv_id,
                })
                return

            is_global = route == Route.GLOBAL
            system, api_messages = build_chat_messages(messages, reranked, route)

            parts: list[str] = []
            provider_used: str | None = None
            model_used: str | None = None
            try:
                for kind, payload in engine.stream(
                    system,
                    api_messages,
                    provider=None if provider in (None, "auto") else provider,
                    max_tokens=settings.answer_max_tokens_global if is_global else None,
                    temperature=settings.answer_temperature_global if is_global else None,
                ):
                    if kind == "provider":
                        provider_used = payload["provider"]
                        model_used = payload["model"]
                        yield format_sse("provider", payload)
                    elif kind == "delta":
                        parts.append(payload)
                        yield format_sse("delta", {"text": payload})
            except ProviderError as exc:
                logger.warning(
                    "chat stream: provider %s failed (%s)", exc.provider, exc.reason
                )
                yield format_sse("error", {
                    "provider": exc.provider,
                    "reason": exc.reason,
                    "message": _human_error(exc.provider, exc.reason, exc.message),
                    "partial": bool(parts),
                })
                return
            except AllProvidersFailedError as exc:
                logger.warning("chat stream: all providers failed (%s)", exc)
                yield format_sse("error", {
                    "provider": None,
                    "reason": "unavailable",
                    "message": _human_error(None, "unavailable", str(exc)),
                    "partial": bool(parts),
                })
                return

            text = "".join(parts).strip()
            citations, grounded = extract_citations(text, reranked)
            citation_dicts = [c.model_dump() for c in citations]
            # Persist the assistant turn only on success — failed turns are not
            # stored, so the user's message stays last and retry "just works".
            conversations.append_message(
                conv_id, "assistant", text,
                citations=citation_dicts, model=provider_used, grounded=grounded,
            )
            yield format_sse("done", {
                "citations": citation_dicts,
                "grounded": grounded,
                "provider": provider_used,
                "model": model_used,
                "conversation_id": conv_id,
            })
        except Exception as exc:  # noqa: BLE001 - the stream must end with an event
            logger.exception("chat stream failed")
            yield format_sse("error", {
                "provider": None, "reason": "error",
                "message": f"Something went wrong: {exc}", "partial": False,
            })

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
