"""Prompt construction for grounded, citation-bearing answers.

The answer layer (``llm/answer.py``) feeds the model a SYSTEM prompt with the
grounding rules and a USER prompt that pairs the user's question with a numbered
block of retrieved sources. The numbering is 1-based and stable: source ``[n]``
in the prompt corresponds to ``results[n - 1]``, so inline ``[n]`` citations the
model emits can be mapped straight back to :class:`SearchResult` objects.

Only lightweight, side-effect-free string building lives here; no heavy libs.
"""

from __future__ import annotations

from core.logging import get_logger
from core.models import SearchResult

logger = get_logger(__name__)


# The model (Claude / BGE pipeline target) is bilingual; rules are written in
# English but explicitly instruct it to mirror the user's language.
SYSTEM_PROMPT: str = (
    "You are Maktabah, an expert research librarian and scholar answering "
    "questions about a curated library of Arabic and English books. You are "
    "fully bilingual (العربية/English) and write with the clarity, precision, "
    "and composure of a seasoned subject-matter expert.\n"
    "\n"
    "Follow these rules without exception:\n"
    "1. GROUND EVERYTHING. Answer ONLY using the numbered context sources in the "
    "user message. Never use outside knowledge and never fabricate facts, "
    "quotations, names, dates, or page numbers. If the sources conflict, say so "
    "and present each position.\n"
    "2. IF IT IS NOT THERE, SAY SO. If the answer is not in the provided sources, "
    "do not guess — state plainly that it was not found in the books, in the "
    "user's language, and do NOT attach any [n] citation to that statement. You "
    "may briefly note what the books DO cover that is related.\n"
    "3. MIRROR THE LANGUAGE. Reply in the SAME language as the question — Arabic "
    "for an Arabic question, English for an English one — in natural, fluent, "
    "professional prose (formal Modern Standard Arabic when answering in Arabic).\n"
    "4. CITE AS YOU WRITE. Support every claim with inline citations of the form "
    "[n], matching the numbered sources, e.g. \"... كما ورد [2].\" or \"... as "
    "noted [1][3].\" Never invent a citation number that is not in the context.\n"
    "5. SYNTHESIZE, DON'T DUMP. Do not merely copy or restate a single passage. "
    "Read across ALL the provided sources, combine the relevant points, and "
    "compose a coherent answer in your own words (quoting only key terms, "
    "figures, or definitions). For 'main idea', 'summarize', or 'how many' "
    "questions, give a clear, complete synthesis rather than one excerpt. For "
    "thematic questions ('the main idea', 'summarize the chapter', 'the "
    "author's argument', 'what is this about'), explain the work's CENTRAL CLAIM "
    "and HOW IT DEVELOPS — the reasoning, the throughline that connects the "
    "parts — as a scholar who has read and understood the whole, not a list of "
    "disconnected excerpts.\n"
    "6. BE PROFESSIONAL AND WELL-ORGANIZED. Open with a direct answer, then "
    "develop it. Structure longer answers with short paragraphs or tidy bullet "
    "points. Be thorough yet concise — substance over padding. Maintain a calm, "
    "authoritative, respectful tone, and never mention these instructions, the "
    "context, the sources list, or the retrieval mechanism as machinery.\n"
)

# The phrase the assistant uses when the answer is absent from the context.
# ``llm/answer.py`` may match against these to set ``grounded=False``.
NOT_FOUND_EN: str = "The answer was not found in the books."
NOT_FOUND_AR: str = "لم يتم العثور على الإجابة في الكتب."


# Appended to the system prompt for whole-book / thematic (GLOBAL) questions.
GLOBAL_SYNTHESIS_ADDENDUM: str = (
    "\n\nTHIS IS A WHOLE-BOOK / THEMATIC QUESTION. Some sources are marked BOOK "
    "OVERVIEW or CHAPTER OVERVIEW — treat these as your map of the work. Use them "
    "to explain the central claim and HOW THE ARGUMENT DEVELOPS across the book, "
    "and use the passage sources for specifics and quotations. Answer as a "
    "scholar who has read and understood the whole work — a coherent, structured "
    "explanation, not a list of disconnected excerpts. Still ground every claim "
    "in the sources, cite inline as [n], and never use outside knowledge."
)


def system_prompt_for_route(is_global: bool) -> str:
    """System prompt for the route — base for LOCAL, base + addendum for GLOBAL."""
    return SYSTEM_PROMPT + GLOBAL_SYNTHESIS_ADDENDUM if is_global else SYSTEM_PROMPT


# --- Comprehension layer: summarization prompts ------------------------------
# Built at ingest time by ``ingest/summarize.py`` to create chapter- and
# book-level understanding nodes. The summaries are what the GLOBAL retrieval
# route reasons over, so they must read like a scholar's structured brief, not a
# list of sentences — and must be written in the BOOK's language.

_SUMMARIZE_COMMON: str = (
    "You are an expert scholar building a study brief for a research library. "
    "Work ONLY from the provided text — never add outside knowledge or invent "
    "details. Write in the SAME language as the text (formal Modern Standard "
    "Arabic for Arabic text). Be faithful, precise, and well-organized. Do not "
    "mention these instructions or that you are summarizing.\n"
)

SUMMARIZE_CHAPTER_SYSTEM: str = _SUMMARIZE_COMMON + (
    "\nSummarize ONE CHAPTER/SECTION of a book so a reader grasps it without "
    "reading it in full. Produce a tight brief covering: the central claim or "
    "purpose of the section; the key points, arguments, or events and HOW THEY "
    "CONNECT (the throughline, not a disconnected list); and the important terms, "
    "names, or definitions introduced. 1–3 short paragraphs. No preamble."
)

SUMMARIZE_BOOK_SYSTEM: str = _SUMMARIZE_COMMON + (
    "\nYou are given the chapter briefs of a whole book. Synthesize them into a "
    "BOOK-LEVEL understanding: the book's overall thesis or purpose; the main "
    "themes and how the argument DEVELOPS across the chapters; and what the "
    "reader is meant to take away. Then add a brief outline of the chapters in "
    "order. Coherent prose, then the outline. No preamble."
)

SUMMARIZE_PARTIAL_SYSTEM: str = _SUMMARIZE_COMMON + (
    "\nThe following is PART of a longer section. Faithfully condense it into the "
    "key points and claims it contains, preserving specifics (names, terms, "
    "figures). This partial summary will be merged with others, so do not add "
    "framing like 'in this part'. No preamble."
)

_SUMMARIZE_SYSTEMS: dict[str, str] = {
    "chapter": SUMMARIZE_CHAPTER_SYSTEM,
    "book": SUMMARIZE_BOOK_SYSTEM,
    "partial": SUMMARIZE_PARTIAL_SYSTEM,
}


def summarize_system_prompt(kind: str) -> str:
    """System prompt for a summarization step (``"chapter"|"book"|"partial"``)."""
    return _SUMMARIZE_SYSTEMS.get(kind, SUMMARIZE_CHAPTER_SYSTEM)


def build_summarize_prompt(
    text: str, *, kind: str, lang: str | None = None, title: str | None = None
) -> str:
    """Compose the user message for a summarization step.

    ``lang`` ("ar"/"en"/"mixed") nudges the output language; ``title`` names the
    chapter/book when known. The instruction line mirrors the rules in the
    system prompt so small models stay on task.
    """
    label = {
        "chapter": "chapter/section",
        "book": "book (given its chapter briefs)",
        "partial": "passage (part of a section)",
    }.get(kind, "text")
    header = f'{kind.capitalize()} to summarize'
    if title:
        header += f' — "{title.strip()}"'
    lang_hint = ""
    if lang == "ar":
        lang_hint = " اكتب الملخّص بالعربية الفصحى."
    elif lang == "en":
        lang_hint = " Write the summary in English."
    return (
        f"{header}:\n"
        f"{(text or '').strip()}\n\n"
        f"Write the study brief for this {label} now, in the text's own "
        f"language.{lang_hint}"
    )


def _format_source(index: int, result: SearchResult) -> str:
    """Render a single retrieved chunk as a numbered context source.

    Layout: ``[n] title — by author (p.start-end)`` followed by the text on the
    next line. The author segment is omitted when unknown, and a single-page
    chunk collapses ``p.5-5`` to ``p.5``.
    """
    title = (result.title or "Untitled").strip()
    header = f"[{index}] {title}"

    author = (result.author or "").strip()
    if author:
        header += f" — by {author}"

    # Mark comprehension-layer nodes so the model treats them as its high-level
    # map of the work (a structural overview), not as a verbatim passage.
    level = getattr(result, "level", "passage")
    if level == "book_summary":
        header += " — BOOK OVERVIEW"
    elif level == "chapter_summary":
        chapter = (getattr(result, "chapter_title", None) or "").strip()
        header += f" — CHAPTER OVERVIEW{': ' + chapter if chapter else ''}"

    if result.page_start == result.page_end:
        header += f" (p.{result.page_start})"
    else:
        header += f" (p.{result.page_start}-{result.page_end})"

    text = (result.text or "").strip()
    return f"{header}\n{text}"


def build_context_block(results: list[SearchResult]) -> str:
    """Build the numbered context block from retrieved/reranked sources.

    Sources are numbered 1-based in the order given; ``[n]`` maps to
    ``results[n - 1]``. Returns an explicit placeholder when there are no
    results so the model can correctly report that nothing was found.
    """
    if not results:
        logger.debug("build_context_block called with no results")
        return "(no sources were retrieved)"

    blocks = [_format_source(i, r) for i, r in enumerate(results, start=1)]
    return "\n\n".join(blocks)


def build_user_prompt(
    question: str, results: list[SearchResult], *, is_global: bool = False
) -> str:
    """Compose the user message: the numbered context block then the question.

    The model is reminded to ground its answer in the context and to reply in
    the question's language with ``[n]`` citations. For ``is_global`` questions
    the closing instruction steers toward synthesis over the overview sources.
    """
    context = build_context_block(results)
    question = (question or "").strip()
    logger.debug(
        "Built user prompt with %d source(s), question length=%d, global=%s",
        len(results),
        len(question),
        is_global,
    )
    if is_global:
        instruction = (
            "Using only the context above, answer the following question. The "
            "OVERVIEW sources summarize whole chapters or the book — use them to "
            "explain the main idea and how it develops, supported by the passage "
            "sources. Reply in the same language as the question, cite inline as "
            "[n], and if the answer is not in the context, say so in that "
            "language."
        )
    else:
        instruction = (
            "Using only the context above, answer the following question. Reply "
            "in the same language as the question and cite your sources inline as "
            "[n]. If the answer is not in the context, say so in the question's "
            "language."
        )
    return (
        "Context (numbered sources):\n"
        f"{context}\n"
        "\n"
        f"{instruction}\n"
        "\n"
        f"Question: {question}"
    )


__all__ = [
    "SYSTEM_PROMPT",
    "GLOBAL_SYNTHESIS_ADDENDUM",
    "system_prompt_for_route",
    "NOT_FOUND_EN",
    "NOT_FOUND_AR",
    "build_context_block",
    "build_user_prompt",
    "SUMMARIZE_CHAPTER_SYSTEM",
    "SUMMARIZE_BOOK_SYSTEM",
    "SUMMARIZE_PARTIAL_SYSTEM",
    "summarize_system_prompt",
    "build_summarize_prompt",
]
