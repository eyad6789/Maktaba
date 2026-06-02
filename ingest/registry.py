"""SQLite-backed ingestion registry.

Tracks every book the pipeline has seen, keyed by the sha256 of the source
file. Enables dedup (skip already-ingested files) and bookkeeping (status,
chunk counts, errors). Uses only the stdlib ``sqlite3`` module so it is safe to
import on any box.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from core.logging import get_logger
from core.models import BookMeta, IngestStats

logger = get_logger(__name__)

# Status values mirrored across the pipeline.
STATUS_STARTED = "started"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS books (
    file_hash   TEXT PRIMARY KEY,
    book_id     TEXT NOT NULL,
    title       TEXT,
    author      TEXT,
    language    TEXT,
    source_path TEXT,
    num_pages   INTEGER DEFAULT 0,
    num_chunks  INTEGER DEFAULT 0,
    status      TEXT NOT NULL,
    error       TEXT,
    updated_at  TEXT NOT NULL
)
"""

# Number of bytes read per chunk when hashing a file.
_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


class Registry:
    """Persistent record of ingested books backed by SQLite.

    The registry is the dedup gate: :meth:`is_ingested` lets the pipeline skip
    files already embedded into Qdrant, while :meth:`mark_started`,
    :meth:`mark_completed` and :meth:`mark_failed` record progress and outcomes.
    """

    def __init__(self, db_path: Path = settings.registry_db) -> None:
        """Open (and create if missing) the registry database at ``db_path``."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.debug("Registry ready at %s", self.db_path)

    # --- connection / schema --------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection with row access by column name."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create the ``books`` table if it does not already exist."""
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    # --- hashing --------------------------------------------------------------

    @staticmethod
    def compute_hash(path: str | Path) -> str:
        """Return the sha256 hex digest of the file at ``path``.

        Streams the file in 1 MiB chunks so large PDFs do not have to be loaded
        fully into memory.
        """
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(_HASH_CHUNK_SIZE), b""):
                sha.update(block)
        return sha.hexdigest()

    # --- queries --------------------------------------------------------------

    def get(self, file_hash: str) -> dict | None:
        """Return the stored row for ``file_hash`` as a dict, or ``None``."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM books WHERE file_hash = ?", (file_hash,)
            ).fetchone()
        return dict(row) if row is not None else None

    def is_ingested(self, file_hash: str) -> bool:
        """Return ``True`` only if the file is recorded as fully completed."""
        record = self.get(file_hash)
        return bool(record) and record.get("status") == STATUS_COMPLETED

    def list_books(self, *, status: str | None = None) -> list[dict]:
        """Return all book rows (newest first), optionally filtered by status."""
        with self._connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM books ORDER BY updated_at DESC"
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM books WHERE status = ? ORDER BY updated_at DESC",
                    (status,),
                ).fetchall()
        return [dict(row) for row in rows]

    # --- mutations ------------------------------------------------------------

    def mark_started(self, book: BookMeta) -> None:
        """Record (or reset) a book as in-progress.

        Upserts on ``file_hash`` so re-ingesting a previously failed book
        cleanly overwrites its prior state and clears any stored error.
        """
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO books (
                    file_hash, book_id, title, author, language,
                    source_path, num_pages, num_chunks, status, error, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?, NULL, ?)
                ON CONFLICT(file_hash) DO UPDATE SET
                    book_id     = excluded.book_id,
                    title       = excluded.title,
                    author      = excluded.author,
                    language    = excluded.language,
                    source_path = excluded.source_path,
                    num_pages   = excluded.num_pages,
                    num_chunks  = 0,
                    status      = excluded.status,
                    error       = NULL,
                    updated_at  = excluded.updated_at
                """,
                (
                    book.file_hash,
                    book.book_id,
                    book.title,
                    book.author,
                    book.language,
                    book.source_path,
                    book.num_pages,
                    STATUS_STARTED,
                    _now_iso(),
                ),
            )
            conn.commit()
        logger.info("Registry: started book_id=%s title=%r", book.book_id, book.title)

    def mark_completed(self, stats: IngestStats) -> None:
        """Mark the book identified by ``stats.book_id`` as completed.

        Updates page/chunk counts and status. The row must already exist
        (created by :meth:`mark_started`); if it does not, a warning is logged.
        """
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE books SET
                    title      = ?,
                    num_pages  = ?,
                    num_chunks = ?,
                    status     = ?,
                    error      = NULL,
                    updated_at = ?
                WHERE book_id = ?
                """,
                (
                    stats.title,
                    stats.num_pages,
                    stats.num_chunks,
                    STATUS_COMPLETED,
                    _now_iso(),
                    stats.book_id,
                ),
            )
            conn.commit()
            if cur.rowcount == 0:
                logger.warning(
                    "Registry: mark_completed found no row for book_id=%s",
                    stats.book_id,
                )
        logger.info(
            "Registry: completed book_id=%s chunks=%d pages=%d",
            stats.book_id,
            stats.num_chunks,
            stats.num_pages,
        )

    def mark_failed(self, book_id: str, error: str) -> None:
        """Record a failure for ``book_id`` with the given error message."""
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE books SET
                    status     = ?,
                    error      = ?,
                    updated_at = ?
                WHERE book_id = ?
                """,
                (STATUS_FAILED, error, _now_iso(), book_id),
            )
            conn.commit()
            if cur.rowcount == 0:
                logger.warning(
                    "Registry: mark_failed found no row for book_id=%s", book_id
                )
        logger.warning("Registry: failed book_id=%s error=%s", book_id, error)
