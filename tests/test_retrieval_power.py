"""Unit tests for the retrieval power-ups: RRF fusion, query-variant parsing,
overlap-aware context expansion, and the rerank score floor.

CI-safe: only pydantic models and pure pipeline helpers — no qdrant/LLM deps.
"""

from __future__ import annotations

from config import settings
from core.models import SearchResult
from retrieval.expand import parse_variants
from retrieval.pipeline import (
    _apply_score_floor,
    expand_context,
    merge_overlapping,
    rrf_fuse,
)


def _result(chunk_id: str, *, book="b1", index=0, text="", level="passage",
            page_start=1, page_end=1, rerank=None) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.0,
        text=text or f"text of {chunk_id}",
        book_id=book,
        title="Book",
        author=None,
        page_start=page_start,
        page_end=page_end,
        chunk_index=index,
        lang="en",
        rerank_score=rerank,
        level=level,
    )


# -- rrf_fuse -------------------------------------------------------------------


def test_rrf_fuse_promotes_chunks_found_by_multiple_queries() -> None:
    a, b, c = _result("a"), _result("b"), _result("c")
    # "b" is mid-ranked in both lists; "a"/"c" each top one list only.
    fused = rrf_fuse([[a, b], [c, b]], k=60)
    assert [r.chunk_id for r in fused][0] == "b"
    assert {r.chunk_id for r in fused} == {"a", "b", "c"}


def test_rrf_fuse_deduplicates_and_scores() -> None:
    a1, a2 = _result("a"), _result("a")
    fused = rrf_fuse([[a1], [a2]], k=60)
    assert len(fused) == 1
    assert abs(fused[0].score - 2 / 61) < 1e-9


def test_rrf_fuse_single_list_keeps_order() -> None:
    a, b = _result("a"), _result("b")
    assert [r.chunk_id for r in rrf_fuse([[a, b]])] == ["a", "b"]


# -- parse_variants ---------------------------------------------------------------


def test_parse_variants_strips_list_markers_and_quotes() -> None:
    out = parse_variants('1. "first query"\n- second query\n* third', "orig", 3)
    assert out == ["first query", "second query", "third"]


def test_parse_variants_drops_original_and_duplicates() -> None:
    out = parse_variants("What is focus?\nwhat is focus?\nAttention defined",
                         "What is focus?", 3)
    assert out == ["Attention defined"]


def test_parse_variants_caps_count_and_skips_blank() -> None:
    out = parse_variants("\n\nquery one\nquery two\nquery three\nquery four", "orig", 2)
    assert out == ["query one", "query two"]


def test_parse_variants_handles_arabic() -> None:
    out = parse_variants("١. ما هو الانضباط؟\nما معنى التركيز؟", "orig", 3)
    assert "ما معنى التركيز؟" in out


# -- merge_overlapping -------------------------------------------------------------


def test_merge_overlapping_splices_shared_text() -> None:
    shared = "the discipline of focus is built over time"
    a = "Chapter one argues that " + shared
    b = shared + " and grows with practice."
    merged = merge_overlapping(a, b)
    assert merged.count(shared) == 1
    assert merged.startswith("Chapter one") and merged.endswith("practice.")


def test_merge_overlapping_falls_back_to_newline_join() -> None:
    assert merge_overlapping("abc def ghi jkl mno pqr", "completely different text") == (
        "abc def ghi jkl mno pqr\ncompletely different text"
    )


def test_merge_overlapping_ignores_tiny_coincidences() -> None:
    # 1-char shared suffix/prefix is below the 20-char floor -> newline join.
    assert merge_overlapping("ends with x", "x starts here") == "ends with x\nx starts here"


# -- expand_context ----------------------------------------------------------------


class FakeStore:
    """Duck-typed stand-in for QdrantStore.fetch_neighbors."""

    def __init__(self, passages: dict[tuple[str, int], SearchResult]):
        self.passages = passages
        self.calls: list[tuple[str, list[int]]] = []

    def fetch_neighbors(self, book_id: str, chunk_indices: list[int]):
        self.calls.append((book_id, list(chunk_indices)))
        return [
            self.passages[(book_id, i)]
            for i in chunk_indices
            if (book_id, i) in self.passages
        ]


def _corpus() -> FakeStore:
    overlap = "shared overlap text between adjacent chunks here"
    return FakeStore({
        ("b1", 0): _result("n0", index=0, text="start of book. " + overlap, page_start=1, page_end=2),
        ("b1", 1): _result("n1", index=1, text=overlap + " middle continues", page_start=2, page_end=3),
        ("b1", 2): _result("n2", index=2, text="unrelated chunk two", page_start=3, page_end=4),
    })


def test_expand_context_stitches_neighbors_and_widens_pages() -> None:
    store = _corpus()
    hit = _result("n1", index=1, text=store.passages[("b1", 1)].text,
                  page_start=2, page_end=3, rerank=0.9)
    [expanded] = expand_context([hit], store, window=1, max_expand=4)
    assert expanded.chunk_id == "n1"                  # identity/scores preserved
    assert "start of book." in expanded.text          # left neighbour stitched
    assert expanded.text.count("shared overlap text") == 1  # overlap spliced once
    assert expanded.page_start == 1 and expanded.page_end == 4


def test_expand_context_window_zero_is_noop() -> None:
    store = _corpus()
    hit = _result("n1", index=1, rerank=0.9)
    assert expand_context([hit], store, window=0) == [hit]
    assert store.calls == []


def test_expand_context_skips_summaries_and_respects_max_expand() -> None:
    store = _corpus()
    summary = _result("s", index=1_000_000, level="chapter_summary", rerank=0.9)
    far = _result("n9", book="b1", index=9, rerank=0.5)
    out = expand_context([summary, far], store, window=1, max_expand=1)
    assert out[0].text == summary.text                # summaries untouched
    assert out[1].text == far.text                    # beyond max_expand untouched


def test_expand_context_does_not_refetch_already_retrieved_chunks() -> None:
    store = _corpus()
    hit0 = _result("n0", index=0, text=store.passages[("b1", 0)].text, rerank=0.9)
    hit1 = _result("n1", index=1, text=store.passages[("b1", 1)].text, rerank=0.8)
    expand_context([hit0, hit1], store, window=1, max_expand=4)
    fetched = {i for _, idxs in store.calls for i in idxs}
    assert 0 not in fetched and 1 not in fetched      # both already in results


# -- score floor -------------------------------------------------------------------


def test_score_floor_disabled_by_default() -> None:
    assert settings.rerank_min_score == 0.0
    weak = _result("w", rerank=0.001)
    assert _apply_score_floor([weak]) == [weak]


def test_score_floor_drops_weak_results() -> None:
    old = settings.rerank_min_score
    settings.rerank_min_score = 0.5
    try:
        strong, weak = _result("s", rerank=0.9), _result("w", rerank=0.1)
        assert _apply_score_floor([strong, weak]) == [strong]
    finally:
        settings.rerank_min_score = old


# -- defaults ----------------------------------------------------------------------


def test_power_up_config_defaults() -> None:
    assert settings.enable_multi_query is True
    assert settings.multi_query_variants == 2
    assert settings.context_window_chunks == 1
    assert settings.context_expand_max_results == 4
    assert settings.qdrant_ef_search == 128
    assert settings.summary_use_chain is True
