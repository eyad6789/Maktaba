"""Conversation persistence endpoints for the chat UI.

CRUD surface over :class:`core.conversations.ConversationStore`:

* ``GET    /conversations``      — sidebar listing, newest-updated first.
* ``POST   /conversations``      — create (201) with optional title/model/scope.
* ``GET    /conversations/{id}`` — full conversation including ordered messages.
* ``PATCH  /conversations/{id}`` — rename.
* ``DELETE /conversations/{id}`` — delete (messages cascade).

The store itself is constructed during the FastAPI lifespan (see
``api/main.py``) and exposed as ``app.state.conversations``; handlers fetch it
through a 503-guarded accessor in the same style as the other singletons.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException, Request

from core.logging import get_logger
from core.models import (
    ConversationCreate,
    ConversationDetail,
    ConversationListResponse,
    ConversationRename,
    ConversationSummary,
)

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from core.conversations import ConversationStore

logger = get_logger(__name__)

router = APIRouter()


# -- state accessor -------------------------------------------------------------


def _get_conversations(request: Request) -> "ConversationStore":
    store = getattr(request.app.state, "conversations", None)
    if store is None:  # pragma: no cover - defensive; lifespan always sets it
        raise HTTPException(
            status_code=503, detail="Conversation store not initialized"
        )
    return store


# -- endpoints ------------------------------------------------------------------


@router.get("/conversations", response_model=ConversationListResponse)
def list_conversations(request: Request) -> ConversationListResponse:
    """List all conversations (newest-updated first) for the sidebar."""
    store = _get_conversations(request)
    return ConversationListResponse(conversations=store.list())


@router.post("/conversations", response_model=ConversationSummary, status_code=201)
def create_conversation(
    request: Request, req: ConversationCreate
) -> ConversationSummary:
    """Create a conversation with optional title, provider and book scope."""
    store = _get_conversations(request)
    created = store.create(title=req.title, model=req.model, book_ids=req.book_ids)
    return ConversationSummary.model_validate(created)


@router.get(
    "/conversations/{conversation_id}", response_model=ConversationDetail
)
def get_conversation(request: Request, conversation_id: str) -> ConversationDetail:
    """Return one conversation with its ordered messages; 404 when unknown."""
    store = _get_conversations(request)
    detail = store.get(conversation_id)
    if detail is None:
        raise HTTPException(
            status_code=404, detail=f"Unknown conversation: {conversation_id}"
        )
    return ConversationDetail.model_validate(detail)


@router.patch(
    "/conversations/{conversation_id}", response_model=ConversationSummary
)
def rename_conversation(
    request: Request, conversation_id: str, req: ConversationRename
) -> ConversationSummary:
    """Rename a conversation; 404 when unknown."""
    store = _get_conversations(request)
    if not store.rename(conversation_id, req.title):
        raise HTTPException(
            status_code=404, detail=f"Unknown conversation: {conversation_id}"
        )
    detail = store.get(conversation_id)
    if detail is None:  # pragma: no cover - deleted between rename and fetch
        raise HTTPException(
            status_code=404, detail=f"Unknown conversation: {conversation_id}"
        )
    # ConversationSummary validation drops the extra "messages" key from the
    # detail dict, so the rename response stays a lightweight summary.
    return ConversationSummary.model_validate(detail)


@router.delete("/conversations/{conversation_id}")
def delete_conversation(request: Request, conversation_id: str) -> dict[str, bool]:
    """Delete a conversation and its messages (cascade); 404 when unknown."""
    store = _get_conversations(request)
    if not store.delete(conversation_id):
        raise HTTPException(
            status_code=404, detail=f"Unknown conversation: {conversation_id}"
        )
    return {"deleted": True}
