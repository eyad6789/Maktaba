"""Hierarchical-level fields on Chunk / SearchResult (Phase 2).

The comprehension layer adds `level`, `parent_id`, `chapter_title`. They must be
fully back-compatible: every pre-existing construction (no level args) stays
valid and defaults to a plain "passage", so old Qdrant points and existing tests
are unaffected.
"""

from __future__ import annotations

from core import schema
from core.models import Chunk, IngestStats, SearchResult


def _minimal_chunk(**kw) -> Chunk:
    base = dict(
        chunk_id="c1", book_id="b1", title="T", text="hello",
        page_start=1, page_end=2, chunk_index=0,
    )
    base.update(kw)
    return Chunk(**base)


def test_chunk_defaults_to_passage():
    c = _minimal_chunk()
    assert c.level == "passage"
    assert c.parent_id is None
    assert c.chapter_title is None


def test_chunk_accepts_summary_levels():
    c = _minimal_chunk(level="chapter_summary", parent_id="b1:summary:book", chapter_title="Chapter 1")
    assert c.level == "chapter_summary"
    assert c.parent_id == "b1:summary:book"
    assert c.chapter_title == "Chapter 1"


def test_search_result_defaults_to_passage():
    r = SearchResult(
        chunk_id="c1", score=0.5, text="x", book_id="b1", title="T",
        page_start=1, page_end=1, chunk_index=0,
    )
    assert r.level == "passage"
    assert r.parent_id is None
    assert r.chapter_title is None


def test_ingest_stats_tracks_summary_nodes():
    s = IngestStats(book_id="b1", title="T")
    assert s.num_summary_nodes == 0


def test_schema_exposes_level_fields_for_payload_and_index():
    # The generic getattr-based writer relies on these being present.
    for field in ("level", "parent_id", "chapter_title"):
        assert field in schema.PAYLOAD_FIELDS
    # level + parent_id must be filterable.
    assert "level" in schema.INDEXED_PAYLOAD_KEYS
    assert "parent_id" in schema.INDEXED_PAYLOAD_KEYS


def test_payload_fields_are_chunk_attributes():
    # Guards the getattr(chunk, field) contract in QdrantStore.upsert_chunks.
    c = _minimal_chunk()
    for field in schema.PAYLOAD_FIELDS:
        assert hasattr(c, field), f"Chunk missing payload field {field!r}"


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
