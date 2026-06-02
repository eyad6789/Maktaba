"""Single-shot grounded answering over retrieved context.

Takes a question plus reranked :class:`SearchResult` candidates, asks the
configured **local** LLM (via :mod:`llm.engine`) to answer *only* from that
context, then parses the model's ``[n]`` citation markers back to the source
chunks. The backend is offline by default — no cloud API required.
"""

from __future__ import annotations

import re

from config import settings
from core.logging import get_logger
from core.models import Answer, Citation, SearchResult
from llm import engine
from llm.prompts import SYSTEM_PROMPT, build_user_prompt

logger = get_logger(__name__)

# Matches inline citation markers like "[1]", "[12]", or grouped "[1, 3]".
_CITATION_RE = re.compile(r"\[\s*(\d+(?:\s*,\s*\d+)*)\s*\]")

# Phrases (lowercased) that signal the answer is not in the books. Kept broad
# and bilingual so grounded-detection does not depend on exact wording.
_NOT_FOUND_MARKERS: tuple[str, ...] = (
    "not found in the books",
    "not found in the provided",
    "not in the books",
    "not in the provided",
    "no information in the books",
    "no relevant information",
    "could not find",
    "cannot find",
    "isn't covered",
    "is not covered",
    "does not appear in",
    "doesn't appear in",
    "no answer in the provided context",
    "غير موجود في الكتب",
    "غير موجودة في الكتب",
    "لا يوجد في الكتب",
    "لا توجد في الكتب",
    "لم أجد",
    "لم اجد",
    "لا تتوفر",
    "ليس في الكتب",
    "غير متوفر في",
)


def _parse_citation_indices(text: str) -> list[int]:
    """Return cited 1-based source numbers in first-seen order, de-duplicated."""
    ordered: list[int] = []
    seen: set[int] = set()
    for match in _CITATION_RE.finditer(text):
        for token in match.group(1).split(","):
            token = token.strip()
            if not token.isdigit():
                continue
            n = int(token)
            if n not in seen:
                seen.add(n)
                ordered.append(n)
    return ordered


def _map_citations(cited_numbers: list[int], results: list[SearchResult]) -> list[Citation]:
    """Map 1-based ``[n]`` markers to :class:`Citation`, guarding the range."""
    citations: list[Citation] = []
    for n in cited_numbers:
        idx = n - 1
        if idx < 0 or idx >= len(results):
            logger.warning("Ignoring out-of-range citation marker [%d]", n)
            continue
        src = results[idx]
        citations.append(
            Citation(
                title=src.title,
                author=src.author,
                page_start=src.page_start,
                page_end=src.page_end,
                book_id=src.book_id,
            )
        )
    return citations


def _looks_not_found(text: str) -> bool:
    """Heuristic: does the answer say the question is not covered by the books?"""
    lowered = text.lower()
    return any(marker in lowered for marker in _NOT_FOUND_MARKERS)


def answer_question(
    question: str,
    results: list[SearchResult],
    *,
    model: str | None = None,
) -> Answer:
    """Answer ``question`` grounded in ``results`` using the local LLM engine.

    Sends ``SYSTEM_PROMPT`` plus the numbered context as a single user turn and
    parses ``[n]`` markers in the reply back to the source chunks. ``grounded``
    is ``False`` only when the model produced no valid citations *and* its reply
    matches the "not found in the books" phrasing.
    """
    chosen_model = model or engine.active_model_name()

    if not results:
        logger.info("No retrieval results for question; returning ungrounded answer")
        return Answer(
            answer="This information was not found in the books.",
            citations=[],
            sources=[],
            model=chosen_model,
            grounded=False,
        )

    user_prompt = build_user_prompt(question, results)
    logger.info("Answering with %s over %d source(s)", chosen_model, len(results))
    answer_text = engine.complete(
        SYSTEM_PROMPT,
        [{"role": "user", "content": user_prompt}],
        model=model,
    )

    cited_numbers = _parse_citation_indices(answer_text)
    citations = _map_citations(cited_numbers, results)
    grounded = bool(citations) or not _looks_not_found(answer_text)

    logger.info(
        "Answer generated: %d char(s), %d citation(s), grounded=%s",
        len(answer_text), len(citations), grounded,
    )
    return Answer(
        answer=answer_text,
        citations=citations,
        sources=results,
        model=chosen_model,
        grounded=grounded,
    )
