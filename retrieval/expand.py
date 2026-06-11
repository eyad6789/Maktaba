"""Multi-query expansion (RAG-Fusion) — rewrite a question into search variants.

A single embedding of the user's question can miss relevant passages that use
different wording, and — on this bilingual corpus — passages written in the
*other* language entirely. :func:`expand_queries` asks a fast LLM for a couple
of retrieval-oriented rewrites (a same-language paraphrase and a cross-language
translation), which the retrieval pipeline searches alongside the original and
fuses with RRF.

Failure-tolerant by design: any LLM error, empty output, or unparseable reply
returns ``[]`` and retrieval proceeds with the original query alone. The LLM is
reached lazily through :mod:`llm.engine`, keeping this module importable on a
box with no model installed.
"""

from __future__ import annotations

import re

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM = (
    "You rewrite a user's question into alternative search queries for a "
    "bilingual (Arabic/English) document search engine. Produce exactly the "
    "requested number of variants, one per line, no numbering, no quotes, no "
    "explanations. The first variant must be a paraphrase in the SAME language "
    "as the question; the second must be a faithful translation of the question "
    "into the OTHER language (Arabic if the question is English, English if it "
    "is Arabic). Keep each variant a single short search query."
)

# Lines like "1. ...", "1) ...", "- ...", "* ..." — strip the list decoration.
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s*")


def parse_variants(text: str, original: str, max_variants: int) -> list[str]:
    """Extract up to ``max_variants`` clean, novel queries from LLM output.

    Pure function (unit-tested): splits lines, strips list markers/quotes,
    drops empties, near-duplicates of the original, and duplicate variants.
    """
    seen = {original.strip().casefold()}
    variants: list[str] = []
    for raw in (text or "").splitlines():
        line = _LIST_PREFIX.sub("", raw).strip().strip('"“”').strip()
        if not line or len(line) < 3:
            continue
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        variants.append(line)
        if len(variants) >= max_variants:
            break
    return variants


def expand_queries(question: str, *, max_variants: int | None = None) -> list[str]:
    """Return up to ``max_variants`` rewrites of ``question`` (never raises)."""
    limit = max_variants if max_variants is not None else settings.multi_query_variants
    if limit <= 0:
        return []

    from llm import engine  # lazy

    # Through the chain (Gemini-fast) when configured and available; otherwise
    # the local utility model. Either way a failure costs nothing but recall.
    use_chain = settings.expansion_use_chain and settings.llm_backend == "fallback"
    model = None if use_chain else settings.utility_model

    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Write {limit} alternative search queries (one per line):"
    )
    try:
        out = engine.complete(
            _SYSTEM,
            [{"role": "user", "content": user}],
            model=model,
            max_tokens=160,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001 - expansion must never break retrieval
        logger.warning("Query expansion failed (%s); searching original only", exc)
        return []

    variants = parse_variants(out, question, limit)
    if variants:
        logger.info("Expanded query %r -> %s", question, variants)
    return variants
