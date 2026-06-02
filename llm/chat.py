"""Multi-turn chat over the book corpus (offline LLM).

Two responsibilities on top of single-shot :mod:`llm.answer`:

1. :func:`condense_query` — rewrite a follow-up message into a *standalone*
   search query using the conversation so far. Skipped (returns the message
   verbatim) when there is only one user turn, so it costs nothing then.
2. :func:`chat_answer` — answer the latest question grounded in the retrieved
   sources while keeping the prior conversation in the model's context.

Generation goes through :mod:`llm.engine` (a local model by default), so chat
works fully offline.
"""

from __future__ import annotations

from config import settings
from core.logging import get_logger
from core.models import Answer, SearchResult
from llm import engine
from llm.answer import _looks_not_found, _map_citations, _parse_citation_indices
from llm.prompts import SYSTEM_PROMPT, build_user_prompt

logger = get_logger(__name__)

_CONDENSE_SYSTEM = (
    "You rewrite the user's latest message into a single standalone search query "
    "for a document search engine. Resolve pronouns and references using the "
    "conversation. Keep the SAME language as the latest user message. Output ONLY "
    "the query text — no quotes, no explanation, no label."
)

_CHAT_SYSTEM = (
    SYSTEM_PROMPT
    + "\nThis is a multi-turn conversation. Use the earlier turns only to "
    "understand the user's intent; ground every factual claim in the numbered "
    "context provided with the latest question."
)


def _user_turns(messages: list[dict]) -> list[dict]:
    return [m for m in messages if m.get("role") == "user"]


def _render_conversation(messages: list[dict]) -> str:
    lines: list[str] = []
    for m in messages:
        role = "User" if m.get("role") == "user" else "Assistant"
        lines.append(f"{role}: {(m.get('content') or '').strip()}")
    return "\n".join(lines)


def condense_query(messages: list[dict], *, model: str | None = None) -> str:
    """Return a standalone retrieval query for the latest user message.

    With a single user turn (or empty input) returns it unchanged and makes no
    model call. Otherwise asks the LLM to fold the conversation into one
    self-contained query. Any failure falls back to the raw latest message.
    """
    users = _user_turns(messages)
    latest = ((users[-1].get("content") if users else "") or "").strip()
    if len(users) <= 1 or not latest:
        return latest

    try:
        condensed = engine.complete(
            _CONDENSE_SYSTEM,
            [
                {
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{_render_conversation(messages)}\n\n"
                        "Standalone search query for the latest user message:"
                    ),
                }
            ],
            model=model,
            max_tokens=128,
            temperature=0.0,
        ).strip().strip('"').strip()
        if condensed:
            logger.info("Condensed query: %r -> %r", latest, condensed)
            return condensed
    except Exception as exc:  # noqa: BLE001 - never fail a chat over condensing
        logger.warning("Query condensation failed (%s); using raw message", exc)
    return latest


def chat_answer(
    messages: list[dict],
    results: list[SearchResult],
    *,
    model: str | None = None,
) -> Answer:
    """Answer the latest user message grounded in ``results``, with history."""
    chosen_model = model or engine.active_model_name()
    users = _user_turns(messages)
    latest = ((users[-1].get("content") if users else "") or "").strip()

    if not results:
        logger.info("No retrieval results for chat; returning ungrounded answer")
        return Answer(
            answer="This information was not found in the books.",
            citations=[],
            sources=[],
            model=chosen_model,
            grounded=False,
        )

    grounded_prompt = build_user_prompt(latest, results)
    api_messages: list[dict] = []
    replaced_last_user = False
    for i in range(len(messages) - 1, -1, -1):
        m = messages[i]
        content = m.get("content") or ""
        if m.get("role") == "user" and not replaced_last_user:
            content = grounded_prompt
            replaced_last_user = True
        api_messages.append({"role": m.get("role"), "content": content})
    api_messages.reverse()

    logger.info("Chat answering with %s over %d source(s)", chosen_model, len(results))
    answer_text = engine.complete(_CHAT_SYSTEM, api_messages, model=model)

    cited = _parse_citation_indices(answer_text)
    citations = _map_citations(cited, results)
    grounded = bool(citations) or not _looks_not_found(answer_text)

    return Answer(
        answer=answer_text,
        citations=citations,
        sources=results,
        model=chosen_model,
        grounded=grounded,
    )
