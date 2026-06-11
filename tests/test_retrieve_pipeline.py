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


# -- GLOBAL score floor + degrade (Phase C) -------------------------------------


def test_global_floor_drops_weak_summaries_and_skips_their_children(monkeypatch):
    from config import settings

    # FakeReranker scores = |overlap| / (|query tokens| + 1). For the query
    # below (7 tokens): bs1 overlaps {is, about, justice} -> 0.375, while cs1
    # overlaps only {justice} -> 0.125. A floor of 0.2 keeps bs1, drops cs1 —
    # and cs1's children must not be drilled once it is gone.
    monkeypatch.setattr(settings, "rerank_min_score", 0.2)
    store = _full_store()
    out = retrieve_for_route(
        "what is the main idea about justice", None, store, FakeReranker(),
        Route.GLOBAL,
    )
    assert {r.chunk_id for r in out} == {"bs1"}
    assert all(c[0] != "children" for c in store.calls)


def test_global_degrades_to_local_when_all_summaries_under_floor(monkeypatch):
    from config import settings

    # Summaries share no tokens with the query (score 0.0 < floor) while the
    # passage overlaps heavily (0.75 >= floor): the floor empties the summary
    # cohort, the route degrades to LOCAL, and the passage still gets through
    # the same floor. (The floor intentionally applies on the degrade path too.)
    monkeypatch.setattr(settings, "rerank_min_score", 0.3)
    store = FakeStore(
        by_level={
            "passage": [_sr("p1", "justice fairness and the rule of law")],
            "chapter_summary": [
                _sr("cs1", "distant nebula overview",
                    level="chapter_summary", parent_id="bs1"),
            ],
            "book_summary": [
                _sr("bs1", "astronomy stars galaxies", level="book_summary"),
            ],
        },
    )
    out = retrieve_for_route(
        "justice fairness law", None, store, FakeReranker(), Route.GLOBAL,
    )
    assert out, "degrade must still answer from passages"
    assert all(r.level == "passage" for r in out)
    assert ("search", ("passage",)) in store.calls


def test_global_degrade_threads_rerank_top_k(monkeypatch):
    # No summary nodes at all -> GLOBAL degrades to LOCAL, and the caller's
    # rerank_top_k must reach the fallback (it used to be hardcoded to None).
    store = FakeStore(
        by_level={
            "passage": [
                _sr("p1", "justice fairness and the rule of law"),
                _sr("p2", "justice in society and fairness of law"),
            ],
        },
    )
    out = retrieve_for_route(
        "justice fairness law", None, store, FakeReranker(), Route.GLOBAL,
        rerank_top_k=1,
    )
    assert len(out) == 1
    assert out[0].level == "passage"


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
