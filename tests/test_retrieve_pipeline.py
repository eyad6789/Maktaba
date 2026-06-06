"""Route-aware retrieval pipeline (Phase 3).

Exercises retrieve_for_route with a fake store (canned results by level) and the
real token-overlap FakeReranker. Assertions check the *level/role* of what comes
back and that GLOBAL drills into children — not exact ordering (the fake
reranker, like the real one, is level-blind, which is exactly why the pipeline
reranks within cohorts).
"""

from __future__ import annotations

from core.models import SearchResult
from retrieval.pipeline import _merge_keep_order, retrieve_for_route
from retrieval.route import Route
from tests.integration.helpers import FakeReranker


def _sr(cid, text, level="passage", parent_id=None):
    return SearchResult(
        chunk_id=cid, score=0.0, text=text, book_id="b1", title="T",
        page_start=1, page_end=1, chunk_index=0, level=level, parent_id=parent_id,
    )


class FakeStore:
    """Returns canned results filtered by the requested levels; records calls."""

    def __init__(self, by_level, children=None):
        self.by_level = by_level
        self.children = children or {}
        self.calls = []

    def hybrid_search(self, query, top_k=50, book_ids=None, levels=None):
        self.calls.append(("search", tuple(levels) if levels else None))
        out = []
        for lv in (levels or list(self.by_level)):
            out.extend(self.by_level.get(lv, []))
        return out

    def fetch_children(self, parent_ids, limit_per_parent=6):
        self.calls.append(("children", tuple(parent_ids)))
        out = []
        for pid in parent_ids:
            out.extend(self.children.get(pid, []))
        return out


def _full_store():
    return FakeStore(
        by_level={
            "passage": [
                _sr("p1", "justice fairness and the rule of law"),
                _sr("p2", "astronomy stars and distant galaxies"),
            ],
            "chapter_summary": [
                _sr("cs1", "chapter overview of justice fairness and law",
                    level="chapter_summary", parent_id="bs1"),
            ],
            "book_summary": [
                _sr("bs1", "this book is about justice and society",
                    level="book_summary"),
            ],
        },
        children={
            "cs1": [_sr("child1", "justice fairness law specific detail passage",
                        level="passage", parent_id="cs1")],
        },
    )


def test_local_returns_only_passages():
    store = _full_store()
    out = retrieve_for_route(
        "justice fairness law", None, store, FakeReranker(), Route.LOCAL,
    )
    assert out, "LOCAL returned nothing"
    assert all(r.level == "passage" for r in out)
    # It searched the passage level explicitly.
    assert ("search", ("passage",)) in store.calls
    # It never went near summaries or children.
    assert all(c[0] != "children" for c in store.calls)


def test_global_returns_summaries_and_drills_into_children():
    store = _full_store()
    out = retrieve_for_route(
        "what is the main idea about justice", None, store, FakeReranker(), Route.GLOBAL,
    )
    ids = {r.chunk_id for r in out}
    levels = {r.level for r in out}
    # A summary node is present (the "understanding").
    assert levels & {"chapter_summary", "book_summary"}, levels
    # It drilled from the kept chapter summary into its child passage.
    assert ("children", ("cs1",)) in store.calls
    assert "child1" in ids
    # Summary search used the summary levels.
    assert any(
        c == ("search", ("book_summary", "chapter_summary")) for c in store.calls
    ), store.calls


def test_global_falls_back_to_passages_without_a_comprehension_layer():
    # A corpus ingested before Phase 4 has no summary nodes.
    store = FakeStore(by_level={"passage": [_sr("p1", "justice fairness law")]})
    out = retrieve_for_route(
        "main idea of justice", None, store, FakeReranker(), Route.GLOBAL,
    )
    assert out and all(r.level == "passage" for r in out)
    # Fell through to a passage search.
    assert ("search", ("passage",)) in store.calls


def test_merge_keep_order_dedupes_by_chunk_id():
    a = [_sr("x", "one"), _sr("y", "two")]
    b = [_sr("y", "two-dup"), _sr("z", "three")]
    merged = _merge_keep_order(a, b)
    assert [r.chunk_id for r in merged] == ["x", "y", "z"]


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
