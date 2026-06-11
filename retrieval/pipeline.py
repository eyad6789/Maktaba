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

Layered on top of the routed core:

* **Multi-query fusion** (:func:`retrieve`) — the question plus LLM-generated
  variants (paraphrase + cross-language translation) are each searched and the
  candidate lists fused with RRF before reranking. Recall layer.
* **Score floor** — reranked passages under ``settings.rerank_min_score`` are
  dropped, so weak matches never reach the prompt. Precision layer.
* **Small-to-big context expansion** — each kept passage is stitched together
  (overlap-aware) with its neighbouring chunks, so the answer model reads full
  context while retrieval stayed precise. Context layer.

Heavy libraries are reached only through the injected ``store``/``reranker``, so
this module imports cleanly without a GPU or ``qdrant_client``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from config import settings
from core.logging import get_logger
from core.models import Embedding, SearchResult
from ingest.normalize import normalize_text
from retrieval.route import Route

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ingest.embed import BGEM3Embedder
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


# -- multi-query fusion ----------------------------------------------------------


def rrf_fuse(
    result_lists: list[list[SearchResult]], *, k: int = 60
) -> list[SearchResult]:
    """Fuse ranked candidate lists with Reciprocal Rank Fusion.

    Each occurrence of a chunk contributes ``1 / (k + rank + 1)``; chunks found
    by several queries rise. The first list's instance wins payload-wise. The
    fused RRF score is written to ``.score``. Pure function (unit-tested).
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, r in enumerate(results):
            scores[r.chunk_id] = scores.get(r.chunk_id, 0.0) + 1.0 / (k + rank + 1)
            first_seen.setdefault(r.chunk_id, r)

    fused = sorted(first_seen.values(), key=lambda r: scores[r.chunk_id], reverse=True)
    for r in fused:
        r.score = scores[r.chunk_id]
    return fused


def _fused_search(
    embeddings: list[Embedding],
    store: "QdrantStore",
    *,
    top_k: int,
    book_ids: list[str] | None,
    levels: list[str] | None,
) -> list[SearchResult]:
    """Hybrid-search every embedding and RRF-fuse the lists (single = passthrough)."""
    lists = [
        store.hybrid_search(emb, top_k=top_k, book_ids=book_ids, levels=levels)
        for emb in embeddings
    ]
    if len(lists) == 1:
        return lists[0]
    fused = rrf_fuse(lists, k=settings.rrf_k)
    logger.debug(
        "Fused %d query variants: %d unique candidates", len(lists), len(fused)
    )
    return fused[:top_k]


# -- precision floor ---------------------------------------------------------------


def _apply_score_floor(results: list[SearchResult]) -> list[SearchResult]:
    """Drop reranked results below ``settings.rerank_min_score`` (0 = keep all)."""
    floor = settings.rerank_min_score
    if floor <= 0 or not results:
        return results
    kept = [r for r in results if (r.rerank_score or 0.0) >= floor]
    if len(kept) != len(results):
        logger.info(
            "Score floor %.2f dropped %d weak candidate(s)",
            floor,
            len(results) - len(kept),
        )
    return kept


# -- small-to-big context expansion ------------------------------------------------


def merge_overlapping(a: str, b: str, *, max_overlap: int = 800) -> str:
    """Join two adjacent chunk texts, splicing out their shared overlap.

    Consecutive chunks share ~90 tokens by construction; the largest suffix of
    ``a`` that prefixes ``b`` (up to ``max_overlap`` chars) is emitted once.
    Falls back to a newline join when no overlap is found. Pure function.
    """
    limit = min(len(a), len(b), max_overlap)
    for k in range(limit, 19, -1):  # overlaps under 20 chars are coincidence
        if a[-k:] == b[:k]:
            return a + b[k:]
    return a + "\n" + b


def _expand_one(
    result: SearchResult, neighbors: dict[int, SearchResult], *, window: int
) -> SearchResult:
    """Stitch ``result`` together with its contiguous neighbours, if present."""
    run = [result.chunk_index]
    for idx in range(result.chunk_index - 1, result.chunk_index - window - 1, -1):
        if idx in neighbors:
            run.insert(0, idx)
        else:
            break
    for idx in range(result.chunk_index + 1, result.chunk_index + window + 1):
        if idx in neighbors:
            run.append(idx)
        else:
            break
    if run == [result.chunk_index]:
        return result

    pieces = [
        neighbors[i] if i != result.chunk_index else result for i in run
    ]
    text = pieces[0].text
    for piece in pieces[1:]:
        text = merge_overlapping(text, piece.text)
    return result.model_copy(
        update={
            "text": text,
            "page_start": min(p.page_start for p in pieces),
            "page_end": max(p.page_end for p in pieces),
        }
    )


def expand_context(
    results: list[SearchResult],
    store: "QdrantStore",
    *,
    window: int | None = None,
    max_expand: int | None = None,
) -> list[SearchResult]:
    """Widen the best passages with their neighbouring chunks (small-to-big).

    Only the first ``max_expand`` passage-level results are widened (summaries
    are already context-rich; the tail stays lean to bound the prompt). Order,
    ids and scores are preserved — only text/page ranges grow.
    """
    window = settings.context_window_chunks if window is None else window
    max_expand = (
        settings.context_expand_max_results if max_expand is None else max_expand
    )
    if window <= 0 or not results:
        return results

    targets = [r for r in results[:max_expand] if r.level == "passage"]
    if not targets:
        return results

    # Indices already in the result set are NOT refetched as neighbours: their
    # text is in the prompt in its own right, and duplicating it wastes tokens.
    present = {(r.book_id, r.chunk_index) for r in results}
    wanted: dict[str, set[int]] = {}
    for r in targets:
        for d in range(1, window + 1):
            for idx in (r.chunk_index - d, r.chunk_index + d):
                if idx >= 0 and (r.book_id, idx) not in present:
                    wanted.setdefault(r.book_id, set()).add(idx)

    neighbor_map: dict[str, dict[int, SearchResult]] = {}
    for book_id, indices in wanted.items():
        try:
            fetched = store.fetch_neighbors(book_id, sorted(indices))
        except Exception as exc:  # noqa: BLE001 - expansion is best-effort
            logger.warning("Context expansion fetch failed for %s: %s", book_id, exc)
            continue
        neighbor_map.setdefault(book_id, {}).update(
            {n.chunk_index: n for n in fetched}
        )

    target_ids = {r.chunk_id for r in targets}
    expanded: list[SearchResult] = []
    for r in results:
        if r.chunk_id in target_ids and neighbor_map.get(r.book_id):
            expanded.append(_expand_one(r, neighbor_map[r.book_id], window=window))
        else:
            expanded.append(r)
    return expanded


# -- routed retrieval ---------------------------------------------------------------


def _retrieve_local(
    question: str,
    embeddings: list[Embedding],
    store: "QdrantStore",
    reranker: "Reranker",
    *,
    book_ids: list[str] | None,
    rerank_top_k: int | None,
) -> list[SearchResult]:
    candidates = _fused_search(
        embeddings,
        store,
        top_k=settings.search_top_k,
        book_ids=book_ids,
        levels=["passage"],
    )
    keep = rerank_top_k if rerank_top_k and rerank_top_k > 0 else settings.rerank_top_k
    reranked = reranker.rerank(question, candidates, top_k=keep)
    reranked = _apply_score_floor(reranked)
    return expand_context(reranked, store)


def _retrieve_global(
    question: str,
    embeddings: list[Embedding],
    store: "QdrantStore",
    reranker: "Reranker",
    *,
    book_ids: list[str] | None,
    rerank_top_k: int | None = None,
) -> list[SearchResult]:
    summary_candidates = _fused_search(
        embeddings,
        store,
        top_k=settings.global_summary_k,
        book_ids=book_ids,
        levels=_SUMMARY_LEVELS,
    )
    if not summary_candidates:
        # No comprehension layer for these books yet — degrade gracefully to a
        # passage search so GLOBAL questions still get an answer.
        logger.info("GLOBAL->LOCAL degrade: no summary nodes for these books")
        return _retrieve_local(
            question,
            embeddings,
            store,
            reranker,
            book_ids=book_ids,
            rerank_top_k=rerank_top_k,
        )

    top_summaries = reranker.rerank(
        question, summary_candidates, top_k=settings.global_summary_keep
    )
    top_summaries = _apply_score_floor(top_summaries)
    if not top_summaries:
        # Every summary scored under the floor — the comprehension layer has
        # nothing relevant to say, so a passage search serves better than junk.
        logger.info("GLOBAL->LOCAL degrade: all summary nodes under score floor")
        return _retrieve_local(
            question,
            embeddings,
            store,
            reranker,
            book_ids=book_ids,
            rerank_top_k=rerank_top_k,
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
    top_children = _apply_score_floor(top_children)

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
    extra_embeddings: list[Embedding] | None = None,
) -> list[SearchResult]:
    """Retrieve and rerank context for ``question`` according to ``route``.

    Returns the ordered list of :class:`SearchResult` to feed the answer prompt:
    raw passages for ``LOCAL``; summary nodes (first) plus supporting child
    passages for ``GLOBAL``. ``extra_embeddings`` (query-variant embeddings from
    multi-query expansion) are searched alongside and RRF-fused.
    """
    embeddings = [embedding, *(extra_embeddings or [])]
    if route == Route.GLOBAL:
        return _retrieve_global(
            question,
            embeddings,
            store,
            reranker,
            book_ids=book_ids,
            rerank_top_k=rerank_top_k,
        )
    return _retrieve_local(
        question, embeddings, store, reranker, book_ids=book_ids, rerank_top_k=rerank_top_k
    )


def retrieve(
    question: str,
    *,
    embedder: "BGEM3Embedder",
    store: "QdrantStore",
    reranker: "Reranker",
    route: Route,
    book_ids: list[str] | None = None,
    rerank_top_k: int | None = None,
) -> list[SearchResult]:
    """Full retrieval entrypoint: multi-query expansion + routed hybrid search.

    Normalizes the question (same Arabic folding the indexed text received —
    otherwise the sparse/lexical channel misses hamza/maqsura/tatweel spelling
    variants, and the cross-encoder scores a raw query against normalized
    passages), embeds it, optionally expands it into bilingual variants (see
    :mod:`retrieval.expand`) embedding those too, then runs the routed
    retrieval with RRF fusion across the variant lists. Callers pass the RAW
    question — normalization happens here, once, for every consumer. This is
    what the API endpoints call; :func:`retrieve_for_route` remains the
    fusion-free core.
    """
    question = normalize_text(question)
    embedding = embedder.embed_query(question)

    extra: list[Embedding] = []
    if settings.enable_multi_query:
        from retrieval.expand import expand_queries  # lazy: pulls llm.engine

        for variant in expand_queries(question):
            try:
                # The LLM may emit unnormalized Arabic (hamza forms, tatweel):
                # variants must match the index the same way the question does.
                extra.append(embedder.embed_query(normalize_text(variant)))
            except Exception as exc:  # noqa: BLE001 - a variant is never worth a 500
                logger.warning("Could not embed variant %r: %s", variant, exc)

    return retrieve_for_route(
        question,
        embedding,
        store,
        reranker,
        route,
        book_ids=book_ids,
        rerank_top_k=rerank_top_k,
        extra_embeddings=extra or None,
    )
