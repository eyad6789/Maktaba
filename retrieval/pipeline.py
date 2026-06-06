"""Route-aware retrieval over the hierarchical comprehension layer.

Two strategies, selected by :class:`retrieval.route.Route`:

* **LOCAL** — search raw passages only, rerank, return. Identical in spirit to
  the original flat retrieval; factual lookups never see summaries.
* **GLOBAL** — search chapter + book *summary* nodes, rerank them, then drill
  into the kept chapters' child passages and rerank those, and finally merge
  (summaries first). The model thus reads the book's structure — its
  "understanding" — together with the supporting detail it cites.

Reranking happens **within each level cohort**, and cohorts are merged by role.
The cross-encoder is level-blind (it scores raw ``(query, text)`` pairs), so a
long summary would otherwise out-score the one precise passage. Keeping the
cohorts separate preserves both the overview and the specifics.

Heavy libraries are reached only through the injected ``store``/``reranker``, so
this module imports cleanly without a GPU or ``qdrant_client``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings
from core.logging import get_logger
from core.models import Embedding, SearchResult
from retrieval.route import Route

if TYPE_CHECKING:  # pragma: no cover - typing only
    from retrieval.rerank import Reranker
    from retrieval.search import QdrantStore

logger = get_logger(__name__)

_SUMMARY_LEVELS = ["book_summary", "chapter_summary"]


def _merge_keep_order(*groups: list[SearchResult]) -> list[SearchResult]:
    """Concatenate result groups in order, de-duplicating by ``chunk_id``."""
    seen: set[str] = set()
    merged: list[SearchResult] = []
    for group in groups:
        for r in group:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                merged.append(r)
    return merged


def _retrieve_local(
    question: str,
    embedding: Embedding,
    store: "QdrantStore",
    reranker: "Reranker",
    *,
    book_ids: list[str] | None,
    rerank_top_k: int | None,
) -> list[SearchResult]:
    candidates = store.hybrid_search(
        embedding,
        top_k=settings.search_top_k,
        book_ids=book_ids,
        levels=["passage"],
    )
    keep = rerank_top_k if rerank_top_k and rerank_top_k > 0 else settings.rerank_top_k
    return reranker.rerank(question, candidates, top_k=keep)


def _retrieve_global(
    question: str,
    embedding: Embedding,
    store: "QdrantStore",
    reranker: "Reranker",
    *,
    book_ids: list[str] | None,
) -> list[SearchResult]:
    summary_candidates = store.hybrid_search(
        embedding,
        top_k=settings.global_summary_k,
        book_ids=book_ids,
        levels=_SUMMARY_LEVELS,
    )
    if not summary_candidates:
        # No comprehension layer for these books yet — degrade gracefully to a
        # passage search so GLOBAL questions still get an answer.
        logger.debug("GLOBAL route found no summary nodes; falling back to passages")
        return _retrieve_local(
            question, embedding, store, reranker, book_ids=book_ids, rerank_top_k=None
        )

    top_summaries = reranker.rerank(
        question, summary_candidates, top_k=settings.global_summary_keep
    )

    chapter_parent_ids = [
        s.chunk_id for s in top_summaries if s.level == "chapter_summary"
    ]
    children = (
        store.fetch_children(chapter_parent_ids, limit_per_parent=settings.global_child_keep)
        if chapter_parent_ids
        else []
    )
    top_children = (
        reranker.rerank(question, children, top_k=settings.global_child_keep)
        if children
        else []
    )

    merged = _merge_keep_order(top_summaries, top_children)
    logger.info(
        "GLOBAL retrieval: %d summary node(s) + %d child passage(s)",
        len(top_summaries),
        len(top_children),
    )
    return merged


def retrieve_for_route(
    question: str,
    embedding: Embedding,
    store: "QdrantStore",
    reranker: "Reranker",
    route: Route,
    *,
    book_ids: list[str] | None = None,
    rerank_top_k: int | None = None,
) -> list[SearchResult]:
    """Retrieve and rerank context for ``question`` according to ``route``.

    Returns the ordered list of :class:`SearchResult` to feed the answer prompt:
    raw passages for ``LOCAL``; summary nodes (first) plus supporting child
    passages for ``GLOBAL``.
    """
    if route == Route.GLOBAL:
        return _retrieve_global(question, embedding, store, reranker, book_ids=book_ids)
    return _retrieve_local(
        question, embedding, store, reranker, book_ids=book_ids, rerank_top_k=rerank_top_k
    )
