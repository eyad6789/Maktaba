"""QdrantStore._build_filter: book + level payload filtering (Phase 3).

The filter is built from real `qdrant_client.models` types, so these are skipped
on the minimal CI image (no qdrant-client) and run locally / in the live env.
Constructing the filter needs no server — only the client library's model types.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qdrant_client")

from retrieval.search import QdrantStore  # noqa: E402


def _conditions(flt):
    return list(flt.must) if flt is not None else []


def test_no_constraints_returns_none():
    # Unfiltered search == original behaviour.
    assert QdrantStore._build_filter(None, None) is None
    assert QdrantStore._build_filter([], []) is None


def test_book_ids_only():
    flt = QdrantStore._build_filter(["b1", "b2"], None)
    conds = _conditions(flt)
    assert len(conds) == 1
    assert conds[0].key == "book_id"
    assert list(conds[0].match.any) == ["b1", "b2"]


def test_levels_only():
    flt = QdrantStore._build_filter(None, ["passage"])
    conds = _conditions(flt)
    assert len(conds) == 1
    assert conds[0].key == "level"
    assert list(conds[0].match.any) == ["passage"]


def test_book_and_levels_combined():
    flt = QdrantStore._build_filter(["b1"], ["chapter_summary", "book_summary"])
    keys = {c.key for c in _conditions(flt)}
    assert keys == {"book_id", "level"}


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
