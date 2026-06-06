"""Structure detection for the comprehension layer (Phase 4).

Covers TOC -> chapter sections, the no-TOC whole-book fallback, chunk bucketing,
and the (gated) Arabic/English heading heuristic. The heading regexes run against
NORMALIZED text (see ingest.normalize), so the Arabic cases use folded forms.
"""

from __future__ import annotations

from core.models import Chunk, PageContent, PageKind
from ingest.structure import (
    Section,
    _detect_headings,
    _sections_from_toc,
    assign_chunks_to_sections,
    detect_structure,
)


def _page(n: int, text: str) -> PageContent:
    return PageContent(page_number=n, text=text, kind=PageKind.NATIVE, lang="en")


def _chunk(cid: str, page: int) -> Chunk:
    return Chunk(
        chunk_id=cid, book_id="b1", title="T", text="x",
        page_start=page, page_end=page, chunk_index=0,
    )


def test_toc_yields_top_level_chapter_sections():
    toc = [(1, "Chapter One", 1), (2, "A subsection", 4), (1, "Chapter Two", 10)]
    sections = _sections_from_toc(toc, num_pages=20)
    assert [s.title for s in sections] == ["Chapter One", "Chapter Two"]
    # Spans close at the next chapter; the last runs to the end.
    assert (sections[0].page_start, sections[0].page_end) == (1, 9)
    assert (sections[1].page_start, sections[1].page_end) == (10, 20)


def test_no_toc_falls_back_to_single_whole_book_section():
    sections = detect_structure([_page(1, "text")], toc=[], num_pages=12)
    assert len(sections) == 1
    assert sections[0].page_start == 1 and sections[0].page_end == 12


def test_detect_structure_prefers_toc():
    toc = [(1, "Intro", 1), (1, "Body", 5)]
    sections = detect_structure([_page(1, "x")], toc=toc, num_pages=9)
    assert len(sections) == 2


def test_assign_chunks_buckets_by_page_including_boundaries():
    sections = [
        Section("Ch1", 1, 1, 5),
        Section("Ch2", 1, 6, 10),
    ]
    chunks = [_chunk("a", 2), _chunk("b", 6), _chunk("c", 10), _chunk("d", 1)]
    buckets = assign_chunks_to_sections(chunks, sections)
    assert {c.chunk_id for c in buckets[0]} == {"a", "d"}    # pages 1,2
    assert {c.chunk_id for c in buckets[1]} == {"b", "c"}    # pages 6,10 (boundary)


def test_heading_heuristic_detects_arabic_chapters_in_normalized_form():
    # First non-empty line of each page is inspected. Arabic folded forms.
    pages = [
        _page(1, "الفصل الاول\nمحتوى الفصل الاول هنا"),
        _page(2, "تابع المحتوى"),
        _page(3, "الفصل الثاني\nمحتوى الفصل الثاني"),
    ]
    sections = _detect_headings(pages, num_pages=4)
    assert len(sections) == 2
    assert sections[0].page_start == 1 and sections[1].page_start == 3


def test_heading_heuristic_detects_english_chapters():
    pages = [
        _page(1, "Chapter 1\nThe beginning of the story"),
        _page(2, "more text"),
        _page(3, "Chapter 2\nThe middle"),
    ]
    sections = _detect_headings(pages, num_pages=4)
    assert [s.page_start for s in sections] == [1, 3]


def test_heading_heuristic_needs_two_headings():
    pages = [_page(1, "Introduction\nonly one heading"), _page(2, "plain text")]
    assert _detect_headings(pages, num_pages=2) == []


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
