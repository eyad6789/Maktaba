"""Functional tests for the SQLite ingestion registry (dedup gate)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from core.models import BookMeta, IngestStats
from ingest.registry import Registry


def _tmp_db() -> Path:
    return Path(tempfile.mkdtemp()) / "registry.db"


def _book(file_hash: str = "hash-1", book_id: str = "book-1") -> BookMeta:
    return BookMeta(
        book_id=book_id,
        title="T",
        author="A",
        language="en",
        source_path="x.pdf",
        num_pages=5,
        file_hash=file_hash,
    )


def test_unknown_hash_not_ingested():
    r = Registry(_tmp_db())
    assert r.is_ingested("does-not-exist") is False
    assert r.get("does-not-exist") is None


def test_started_is_not_yet_ingested():
    r = Registry(_tmp_db())
    b = _book()
    r.mark_started(b)
    assert r.is_ingested(b.file_hash) is False  # started != completed


def test_completed_is_ingested_and_counts_persist():
    r = Registry(_tmp_db())
    b = _book()
    r.mark_started(b)
    r.mark_completed(
        IngestStats(book_id=b.book_id, title=b.title, num_pages=5, num_chunks=10, status="completed")
    )
    assert r.is_ingested(b.file_hash) is True
    row = r.get(b.file_hash)
    assert row is not None
    assert row["num_chunks"] == 10
    assert row["status"] == "completed"


def test_failed_status_recorded_and_not_ingested():
    r = Registry(_tmp_db())
    b = _book(file_hash="hash-2", book_id="book-2")
    r.mark_started(b)
    r.mark_failed(b.book_id, "boom")
    assert r.is_ingested(b.file_hash) is False
    assert r.get(b.file_hash)["error"] == "boom"


def test_list_books_returns_all():
    r = Registry(_tmp_db())
    r.mark_started(_book(file_hash="a", book_id="ba"))
    r.mark_started(_book(file_hash="b", book_id="bb"))
    assert len(r.list_books()) == 2


def test_compute_hash_is_stable_sha256():
    fd, path = tempfile.mkstemp()
    try:
        os.write(fd, b"hello world")
    finally:
        os.close(fd)
    h1 = Registry.compute_hash(path)
    h2 = Registry.compute_hash(path)
    assert h1 == h2
    assert len(h1) == 64  # sha256 hex digest


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
