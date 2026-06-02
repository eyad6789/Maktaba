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
    "questions, give a clear, complete synthesis rather than one excerpt.\n"
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


def build_user_prompt(question: str, results: list[SearchResult]) -> str:
    """Compose the user message: the numbered context block then the question.

    The model is reminded to ground its answer in the context and to reply in
    the question's language with ``[n]`` citations.
    """
    context = build_context_block(results)
    question = (question or "").strip()
    logger.debug(
        "Built user prompt with %d source(s), question length=%d",
        len(results),
        len(question),
    )
    return (
        "Context (numbered sources):\n"
        f"{context}\n"
        "\n"
        "Using only the context above, answer the following question. Reply in "
        "the same language as the question and cite your sources inline as [n]. "
        "If the answer is not in the context, say so in the question's "
        "language.\n"
        "\n"
        f"Question: {question}"
    )


__all__ = [
    "SYSTEM_PROMPT",
    "NOT_FOUND_EN",
    "NOT_FOUND_AR",
    "build_context_block",
    "build_user_prompt",
]
