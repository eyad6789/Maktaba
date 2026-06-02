"""Functional tests for page-aware chunking."""

from __future__ import annotations

from core.models import BookMeta, PageContent, PageKind
from ingest.chunk import chunk_pages


def _book() -> BookMeta:
    return BookMeta(
        book_id="book-1",
        title="Test Book",
        author="Author",
        source_path="x.pdf",
        file_hash="deadbeef",
        num_pages=2,
    )


def _pages() -> list[PageContent]:
    p1 = "Sentence one here. Sentence two here. " * 40
    p2 = "Second page sentence. Another one follows here. " * 40
    return [
        PageContent(page_number=1, text=p1, kind=PageKind.NATIVE, lang="en"),
        PageContent(page_number=2, text=p2, kind=PageKind.NATIVE, lang="en"),
    ]


def test_produces_multiple_chunks():
    chunks = chunk_pages(_pages(), _book(), target_tokens=80, overlap_tokens=15, min_tokens=10)
    assert len(chunks) >= 2


def test_sequential_index_and_unique_ids():
    chunks = chunk_pages(_pages(), _book(), target_tokens=80, overlap_tokens=15, min_tokens=10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    ids = [c.chunk_id for c in chunks]
    assert len(set(ids)) == len(ids)


def test_page_spans_valid_and_cover_both_pages():
    chunks = chunk_pages(_pages(), _book(), target_tokens=80, overlap_tokens=15, min_tokens=10)
    for c in chunks:
        assert 1 <= c.page_start <= c.page_end <= 2
    assert any(c.page_start == 1 for c in chunks)
    assert any(c.page_end == 2 for c in chunks)


def test_chunk_ids_stable_across_runs():
    a = chunk_pages(_pages(), _book(), target_tokens=80, overlap_tokens=15, min_tokens=10)
    b = chunk_pages(_pages(), _book(), target_tokens=80, overlap_tokens=15, min_tokens=10)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]


def test_text_is_populated_and_metadata_propagated():
    chunks = chunk_pages(_pages(), _book(), target_tokens=80, overlap_tokens=15, min_tokens=10)
    for c in chunks:
        assert c.text.strip()
        assert c.book_id == "book-1"
        assert c.title == "Test Book"
        assert c.token_count > 0


def test_empty_input_yields_no_chunks():
    assert chunk_pages([], _book()) == []
    blank = [PageContent(page_number=1, text="   ", kind=PageKind.NATIVE, lang="en")]
    assert chunk_pages(blank, _book()) == []


def test_single_small_page_kept_as_only_chunk():
    pages = [PageContent(page_number=1, text="Tiny page.", kind=PageKind.NATIVE, lang="en")]
    chunks = chunk_pages(pages, _book(), target_tokens=600, overlap_tokens=50, min_tokens=50)
    assert len(chunks) == 1  # kept despite being below min_tokens (only chunk)


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
