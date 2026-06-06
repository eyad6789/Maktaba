"""Query routing — choose the retrieval/answer strategy for a question.

A cheap, bilingual heuristic decides whether a question is a whole-book/thematic
GLOBAL question (reason over chapter+book summaries) or a specific factual LOCAL
lookup (raw passages). The heuristic makes NO model call, so it is free on the
hot path; an optional small-LLM fallback (off by default) breaks ties only when
``settings.router_use_llm`` is set. Everything here is lightweight to import.
"""

from __future__ import annotations

from enum import Enum

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)


class Route(str, Enum):
    """How to answer a question over the hierarchical book index.

    * ``LOCAL`` — a specific factual lookup. Retrieve raw passages only.
    * ``GLOBAL`` — a whole-book / thematic / comparative question. Reason over
      chapter + book summary nodes, then drill into supporting passages.
    """

    LOCAL = "local"
    GLOBAL = "global"


# Whole-book / thematic / comparative cues -> GLOBAL. Checked first so phrases
# like "what is the main idea" route GLOBAL despite containing "what is".
_GLOBAL_CUES: tuple[str, ...] = (
    # English
    "main idea", "main ideas", "main point", "main theme", "central idea",
    "central theme", "central argument", "key idea", "key theme", "overall",
    "in general", "summary", "summarize", "summarise", "overview", "gist",
    "thesis", "the argument", "author's argument", "argument of", "the theme",
    "themes", "what is this book about", "what is the book about",
    "what is it about", "what's it about", "about the book", "the book about",
    "purpose of", "point of the", "takeaway", "take away", "conclusion of",
    "how does", "how do", "develop", "compare", "comparison", "contrast",
    "difference between", "differences between", "relationship between",
    "explain the book", "tell me about the book",
    # Arabic
    "الفكرة الرئيسية", "الافكار الرئيسية", "الأفكار الرئيسية", "الفكرة الأساسية",
    "الفكرة الاساسية", "الموضوع الرئيسي", "الفكرة العامة", "بشكل عام", "بشكل عامّ",
    "لخص", "لخّص", "تلخيص", "ملخص", "ملخّص", "نظرة عامة", "خلاصة", "الخلاصة",
    "ما موضوع", "ما هو موضوع", "موضوع الكتاب", "فكرة الكتاب", "عمّ يتحدث",
    "عما يتحدث", "الهدف من", "الأطروحة", "حجة المؤلف", "قارن", "المقارنة",
    "الفرق بين", "العلاقة بين", "كيف يتطور", "كيف تتطور", "الفكره الرئيسيه",
)

# Specific factual cues -> LOCAL.
_LOCAL_CUES: tuple[str, ...] = (
    # English
    "how many", "how much", "what year", "what date", "when did", "when was",
    "who is", "who was", "who were", "where is", "where did", "define",
    "definition of", "what is the meaning", "meaning of", "which page",
    "on page", "list the", "name the", "give an example",
    # Arabic
    "كم عدد", "كم عدد", "كم", "متى", "في أي عام", "في اي عام", "من هو", "من هي",
    "أين", "اين", "عرّف", "عرف", "تعريف", "ما معنى", "معنى", "في أي صفحة",
    "في اي صفحة", "صفحة",
)


def heuristic_route(question: str, book_ids: list[str] | None = None) -> "Route | None":
    """Route from surface cues, or ``None`` when the question is ambiguous.

    GLOBAL cues win over LOCAL cues (thematic phrasing often embeds factual
    words). Returns ``None`` when no cue matches, so the caller can decide a
    default (or invoke the optional LLM fallback).
    """
    q = (question or "").lower()
    if any(cue in q for cue in _GLOBAL_CUES):
        return Route.GLOBAL
    if any(cue in q for cue in _LOCAL_CUES):
        return Route.LOCAL
    return None


def _llm_route(question: str) -> "Route | None":
    """Optional small-LLM tie-breaker. Best-effort; never raises."""
    try:
        from llm import engine  # lazy — only when router_use_llm is set

        out = engine.complete(
            "Classify the user's question about a book as exactly one word: "
            "GLOBAL (about the whole book / a chapter / its main ideas, themes, "
            "argument, or a comparison) or LOCAL (a specific factual lookup). "
            "Answer with only GLOBAL or LOCAL.",
            [{"role": "user", "content": question}],
            model=settings.utility_model,
            max_tokens=4,
            temperature=0.0,
        )
        return Route.GLOBAL if "global" in (out or "").lower() else Route.LOCAL
    except Exception as exc:  # noqa: BLE001 - routing must never fail a query
        logger.warning("LLM route fallback failed (%s); defaulting to LOCAL", exc)
        return None


def coerce_route(value: "str | Route | None") -> "Route | None":
    """Parse an explicit route override (e.g. from the API) into a ``Route``."""
    if value is None or value == "":
        return None
    if isinstance(value, Route):
        return value
    try:
        return Route(str(value).strip().lower())
    except ValueError:
        logger.warning("Unknown route override %r; ignoring", value)
        return None


def classify_route(question: str, book_ids: list[str] | None = None) -> Route:
    """Pick a route for ``question``: heuristic, optional LLM tie-break, else LOCAL.

    LOCAL is the safe default — it preserves the original precise-passage
    behaviour for anything the heuristic cannot confidently call GLOBAL.
    """
    route = heuristic_route(question, book_ids)
    if route is not None:
        return route
    if settings.router_use_llm:
        route = _llm_route(question)
        if route is not None:
            return route
    return Route.LOCAL
