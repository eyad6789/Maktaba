"""Summary-node construction for the comprehension layer (Phase 4).

The LLM is faked (no model/server). We verify node shape, level/parent links,
deterministic ids, the single-section short-circuit, and that a long section
triggers map/reduce (more than one LLM call).
"""

from __future__ import annotations

import llm.engine as engine_mod
from config import settings
from core.models import BookMeta, Chunk
from ingest.structure import Section
from ingest.summarize import build_summary_nodes, summarize_section


def _book(language="en", num_pages=10) -> BookMeta:
    return BookMeta(
        book_id="b1", title="Test Book", author="A", language=language,
        source_path="x.pdf", file_hash="h", num_pages=num_pages,
    )


def _chunk(cid, page, text="some text", tokens=400) -> Chunk:
    return Chunk(
        chunk_id=cid, book_id="b1", title="Test Book", text=text,
        page_start=page, page_end=page, chunk_index=0, lang="en", token_count=tokens,
    )


def _with_fake_engine(fn):
    """Run fn() with engine.complete stubbed, restoring it afterwards."""
    orig = engine_mod.complete
    try:
        return fn()
    finally:
        engine_mod.complete = orig


def test_multi_section_builds_chapter_and_book_nodes():
    def body():
        engine_mod.complete = lambda system, messages, **kw: "SUMMARY"
        sections = [Section("Chapter One", 1, 1, 5), Section("Chapter Two", 1, 6, 10)]
        chunks = [_chunk("a", 2), _chunk("b", 7)]
        nodes = build_summary_nodes(sections, chunks, _book())
        levels = [n.level for n in nodes]
        assert levels.count("chapter_summary") == 2
        assert levels.count("book_summary") == 1
        chapters = [n for n in nodes if n.level == "chapter_summary"]
        book = next(n for n in nodes if n.level == "book_summary")
        # Chapter nodes link to the book node; titles propagate.
        assert all(c.parent_id == book.chunk_id for c in chapters)
        assert {c.chapter_title for c in chapters} == {"Chapter One", "Chapter Two"}
        assert book.parent_id is None
        # Book node spans the whole book.
        assert book.page_start == 1 and book.page_end == 10

    _with_fake_engine(body)


def test_single_section_yields_only_a_book_node():
    def body():
        engine_mod.complete = lambda system, messages, **kw: "SUMMARY"
        sections = [Section("(whole book)", 0, 1, 10)]
        nodes = build_summary_nodes(sections, [_chunk("a", 1)], _book())
        assert len(nodes) == 1 and nodes[0].level == "book_summary"

    _with_fake_engine(body)


def test_summary_node_ids_are_deterministic():
    def body():
        engine_mod.complete = lambda system, messages, **kw: "SUMMARY"
        sections = [Section("Ch1", 1, 1, 5), Section("Ch2", 1, 6, 10)]
        chunks = [_chunk("a", 2), _chunk("b", 7)]
        ids1 = [n.chunk_id for n in build_summary_nodes(sections, chunks, _book())]
        ids2 = [n.chunk_id for n in build_summary_nodes(sections, chunks, _book())]
        assert ids1 == ids2
        assert len(set(ids1)) == len(ids1)  # disjoint

    _with_fake_engine(body)


def test_no_chunks_yields_no_nodes():
    def body():
        engine_mod.complete = lambda system, messages, **kw: "SUMMARY"
        assert build_summary_nodes([Section("x", 0, 1, 1)], [], _book()) == []

    _with_fake_engine(body)


def test_long_section_triggers_map_reduce():
    def body():
        calls = {"n": 0}

        def counting(system, messages, **kw):
            calls["n"] += 1
            return "PARTIAL"

        engine_mod.complete = counting
        orig_batch = settings.summary_map_batch_tokens
        settings.summary_map_batch_tokens = 5  # force batching
        try:
            section = Section("Long Chapter", 1, 1, 5)
            long_chunks = [
                _chunk("a", 1, text="word " * 200, tokens=400),
                _chunk("b", 2, text="word " * 200, tokens=400),
            ]
            node = summarize_section(
                section, long_chunks, _book(), index=0, parent_id="b1:summary:book",
            )
        finally:
            settings.summary_map_batch_tokens = orig_batch
        assert node.level == "chapter_summary"
        # >1 call means map (partials) + reduce happened, not a single shot.
        assert calls["n"] > 1

    _with_fake_engine(body)


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
