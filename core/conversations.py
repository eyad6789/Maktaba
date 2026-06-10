"""SQLite-backed conversation store for the chat UI.

Persists chat conversations and their messages (content, citations, the
answering provider and groundedness) so the web app can show a ChatGPT-style
sidebar of past conversations. Lives in its own database file next to the
ingestion registry (``settings.conversations_db`` — registry.db is never
touched) and uses only the stdlib ``sqlite3`` module so it is safe to import
on any box.

All methods return plain dicts shaped exactly like the conversation models in
``core/models.py`` (``ConversationSummary`` / ``ConversationDetail`` /
``ConversationMessage``) so API routes can validate them directly.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)

# Auto-titles (derived from the first user message) are capped at this many
# characters, the trailing ellipsis included.
_TITLE_MAX_CHARS = 60

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT '',
    model       TEXT NOT NULL DEFAULT 'auto',
    book_ids    TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    citations       TEXT,
    model           TEXT,
    grounded        INTEGER,
    created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, id);
"""


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _dump_json(value: list | None) -> str | None:
    """Serialize ``value`` for a JSON text column; ``None`` stays NULL.

    ``ensure_ascii=False`` keeps Arabic titles/citations human-readable when
    inspecting the database directly.
    """
    return None if value is None else json.dumps(value, ensure_ascii=False)


def _load_json(text: str | None) -> list | None:
    """Decode a JSON text column; NULL stays ``None``."""
    return None if text is None else json.loads(text)


def _make_title(text: str) -> str:
    """Trim ``text`` into a sidebar title.

    Collapses whitespace, then truncates to at most ``_TITLE_MAX_CHARS``
    characters on a word boundary, appending a single ellipsis character when
    truncated. Pure string slicing — works the same for Arabic and English and
    never calls an LLM.
    """
    title = " ".join(text.split())
    if len(title) <= _TITLE_MAX_CHARS:
        return title
    cut = title[: _TITLE_MAX_CHARS - 1]  # leave room for the ellipsis
    space = cut.rfind(" ")
    if space > 0:  # back off to the last complete word
        cut = cut[:space]
    return cut.rstrip() + "…"


def _summary_dict(row: sqlite3.Row, message_count: int) -> dict:
    """Convert a ``conversations`` row into a ``ConversationSummary`` dict."""
    return {
        "id": row["id"],
        "title": row["title"],
        "model": row["model"],
        "book_ids": _load_json(row["book_ids"]),
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "message_count": message_count,
    }


def _message_dict(row: sqlite3.Row) -> dict:
    """Convert a ``messages`` row into a ``ConversationMessage`` dict."""
    return {
        "id": row["id"],
        "role": row["role"],
        "content": row["content"],
        "citations": _load_json(row["citations"]),
        "model": row["model"],
        "grounded": None if row["grounded"] is None else bool(row["grounded"]),
        "created_at": row["created_at"],
    }


class ConversationStore:
    """Persistent chat history backed by SQLite.

    Conversations carry per-conversation defaults (provider ``model``, book
    scope ``book_ids``); messages carry per-turn results (citations, answering
    provider, groundedness). Deleting a conversation cascades to its messages.
    Sidebar ordering is "most recently updated first", where ``updated_at`` is
    bumped on every appended message (rename does not reorder).
    """

    def __init__(self, db_path: Path | str = settings.conversations_db) -> None:
        """Open (and create if missing) the conversations database at ``db_path``."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        logger.debug("ConversationStore ready at %s", self.db_path)

    # --- connection / schema --------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        """Open a new connection with row access by column name.

        WAL keeps reads (sidebar listing) from blocking writes (streamed turns
        being appended); ``foreign_keys`` must be re-enabled per connection for
        the messages ON DELETE CASCADE to fire.
        """
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Create the ``conversations``/``messages`` tables if missing."""
        with self._connect() as conn:
            conn.executescript(_SCHEMA)
            conn.commit()

    # --- queries --------------------------------------------------------------

    def list(self) -> list[dict]:
        """Return all conversations as summary dicts, newest-updated first.

        Each dict includes ``message_count`` so the sidebar can render without
        a per-conversation round trip.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT c.*, COUNT(m.id) AS message_count
                FROM conversations AS c
                LEFT JOIN messages AS m ON m.conversation_id = c.id
                GROUP BY c.id
                ORDER BY c.updated_at DESC, c.rowid DESC
                """
            ).fetchall()
        return [_summary_dict(row, row["message_count"]) for row in rows]

    def get(self, conv_id: str) -> dict | None:
        """Return one conversation as a detail dict, or ``None`` if unknown.

        The dict includes the ordered ``messages`` list (citations decoded from
        JSON, ``grounded`` as ``bool | None``).
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            if row is None:
                return None
            msg_rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY id",
                (conv_id,),
            ).fetchall()
        detail = _summary_dict(row, len(msg_rows))
        detail["messages"] = [_message_dict(m) for m in msg_rows]
        return detail

    # --- mutations ------------------------------------------------------------

    def create(
        self,
        title: str = "",
        model: str = "auto",
        book_ids: list[str] | None = None,
    ) -> dict:
        """Create a conversation and return its summary dict.

        ``title`` may stay empty — :meth:`set_title_from_message` fills it in
        from the first user message. ``book_ids=None`` means "all books".
        """
        conv_id = uuid.uuid4().hex
        now = _now_iso()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, title, model, book_ids, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conv_id, title, model, _dump_json(book_ids), now, now),
            )
            conn.commit()
        logger.info("Conversations: created id=%s model=%s", conv_id, model)
        return {
            "id": conv_id,
            "title": title,
            "model": model,
            "book_ids": book_ids,
            "created_at": now,
            "updated_at": now,
            "message_count": 0,
        }

    def append_message(
        self,
        conv_id: str,
        role: str,
        content: str,
        *,
        citations: list[dict] | None = None,
        model: str | None = None,
        grounded: bool | None = None,
    ) -> dict:
        """Append one turn to a conversation and return its message dict.

        ``citations`` accepts a list of plain dicts (``Citation.model_dump()``)
        and is stored as JSON. Bumps the conversation's ``updated_at`` so the
        sidebar reorders by activity.

        Raises ``KeyError`` when ``conv_id`` does not exist.
        """
        now = _now_iso()
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM conversations WHERE id = ?", (conv_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown conversation: {conv_id}")
            cur = conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, citations, model, grounded, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conv_id,
                    role,
                    content,
                    _dump_json(citations),
                    model,
                    None if grounded is None else int(bool(grounded)),
                    now,
                ),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conv_id),
            )
            conn.commit()
            message_id = cur.lastrowid
        return {
            "id": message_id,
            "role": role,
            "content": content,
            "citations": citations,
            "model": model,
            "grounded": None if grounded is None else bool(grounded),
            "created_at": now,
        }

    def rename(self, conv_id: str, title: str) -> bool:
        """Set a conversation's title; ``False`` when ``conv_id`` is unknown.

        Deliberately does not bump ``updated_at`` — renaming should not move a
        conversation to the top of the sidebar.
        """
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ?",
                (title, conv_id),
            )
            conn.commit()
        return cur.rowcount > 0

    def delete(self, conv_id: str) -> bool:
        """Delete a conversation (messages cascade); ``False`` when unknown."""
        with self._connect() as conn:
            cur = conn.execute(
                "DELETE FROM conversations WHERE id = ?", (conv_id,)
            )
            conn.commit()
        deleted = cur.rowcount > 0
        if deleted:
            logger.info("Conversations: deleted id=%s", conv_id)
        return deleted

    def set_title_from_message(self, conv_id: str, text: str) -> str | None:
        """Auto-title a conversation from its first user message.

        Only sets the title when the current title is empty, so explicit and
        user-renamed titles are never overwritten. The title is ``text``
        trimmed to at most 60 characters on a word boundary with a trailing
        ellipsis character when truncated (see :func:`_make_title`).

        Returns the title that was set, or ``None`` when nothing changed
        (unknown conversation, non-empty existing title, or blank ``text``).
        """
        title = _make_title(text)
        if not title:
            return None
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE conversations SET title = ? WHERE id = ? AND title = ''",
                (title, conv_id),
            )
            conn.commit()
        return title if cur.rowcount > 0 else None
