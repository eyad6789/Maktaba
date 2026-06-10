"""Functional tests for the SQLite conversation store (chat persistence).

CI-safe: stdlib + pydantic only — no fastapi/openai/anthropic/qdrant imports.
The returned dicts are validated against the conversation models in
``core/models.py`` to prove the store and the API routes agree on shapes.
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path

from core.conversations import ConversationStore
from core.models import (
    Citation,
    ConversationDetail,
    ConversationListResponse,
    ConversationSummary,
)


def _tmp_db() -> Path:
    return Path(tempfile.mkdtemp()) / "conversations.db"


def _citations() -> list[dict]:
    """Citation payloads exactly as the answer layer persists them."""
    return [
        Citation(
            title="Kitab", author="Author", page_start=10, page_end=12, book_id="b1"
        ).model_dump(),
        Citation(
            title="كتاب الفقه", author=None, page_start=3, page_end=3, book_id="b2"
        ).model_dump(),
    ]


def test_create_list_get_round_trip_and_shapes():
    s = ConversationStore(_tmp_db())
    created = s.create(title="My chat", model="gemini", book_ids=["b1", "b2"])

    summary = ConversationSummary.model_validate(created)
    assert summary.id
    assert summary.title == "My chat"
    assert summary.model == "gemini"
    assert summary.book_ids == ["b1", "b2"]
    assert summary.message_count == 0
    assert summary.created_at == summary.updated_at

    listing = ConversationListResponse.model_validate({"conversations": s.list()})
    assert [c.id for c in listing.conversations] == [summary.id]
    assert listing.conversations[0].book_ids == ["b1", "b2"]

    detail = ConversationDetail.model_validate(s.get(summary.id))
    assert detail.id == summary.id
    assert detail.messages == []


def test_create_defaults_empty_title_auto_model_all_books():
    s = ConversationStore(_tmp_db())
    c = s.create()
    assert c["title"] == ""
    assert c["model"] == "auto"
    assert c["book_ids"] is None  # None = all books
    assert s.get(c["id"])["book_ids"] is None  # NULL round-trips, not "null"


def test_list_orders_newest_updated_first():
    s = ConversationStore(_tmp_db())
    a = s.create(title="a")
    time.sleep(0.002)  # guarantee distinct ISO timestamps
    b = s.create(title="b")
    assert [c["id"] for c in s.list()] == [b["id"], a["id"]]

    # Appending to the older conversation bumps it back to the top.
    time.sleep(0.002)
    s.append_message(a["id"], "user", "hello")
    assert [c["id"] for c in s.list()] == [a["id"], b["id"]]
    rows = {c["id"]: c for c in s.list()}
    assert rows[a["id"]]["updated_at"] > rows[a["id"]]["created_at"]


def test_list_includes_message_count():
    s = ConversationStore(_tmp_db())
    a = s.create()
    b = s.create()
    s.append_message(a["id"], "user", "q")
    s.append_message(a["id"], "assistant", "a")
    counts = {c["id"]: c["message_count"] for c in s.list()}
    assert counts == {a["id"]: 2, b["id"]: 0}
    assert s.get(a["id"])["message_count"] == 2


def test_append_message_citations_round_trip():
    s = ConversationStore(_tmp_db())
    c = s.create()
    cites = _citations()
    msg = s.append_message(
        c["id"], "assistant", "answer [1][2]",
        citations=cites, model="gemini", grounded=True,
    )
    assert msg["citations"] == cites
    assert isinstance(msg["id"], int)

    detail = ConversationDetail.model_validate(s.get(c["id"]))
    stored = detail.messages[0]
    assert [ct.model_dump() for ct in stored.citations] == cites  # incl. Arabic title
    assert stored.model == "gemini"
    assert stored.grounded is True
    assert stored.content == "answer [1][2]"


def test_messages_ordered_and_grounded_round_trips_as_bool_or_none():
    s = ConversationStore(_tmp_db())
    c = s.create()
    s.append_message(c["id"], "user", "q")  # no citations/model/grounded
    s.append_message(c["id"], "assistant", "not found", model="local", grounded=False)
    s.append_message(c["id"], "assistant", "found", grounded=True)
    msgs = s.get(c["id"])["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "assistant"]
    assert [m["grounded"] for m in msgs] == [None, False, True]
    assert msgs[0]["citations"] is None
    assert [m["id"] for m in msgs] == sorted(m["id"] for m in msgs)


def test_delete_cascades_to_messages():
    db = _tmp_db()
    s = ConversationStore(db)
    c = s.create()
    s.append_message(c["id"], "user", "hello")
    s.append_message(c["id"], "assistant", "hi")
    assert s.delete(c["id"]) is True
    assert s.get(c["id"]) is None

    # The ON DELETE CASCADE must remove the orphaned message rows too.
    conn = sqlite3.connect(str(db))
    try:
        remaining = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    finally:
        conn.close()
    assert remaining == 0


def test_rename_updates_title():
    s = ConversationStore(_tmp_db())
    c = s.create(title="old")
    assert s.rename(c["id"], "new title") is True
    assert s.get(c["id"])["title"] == "new title"


def test_auto_title_truncates_on_word_boundary_english():
    s = ConversationStore(_tmp_db())
    c = s.create()
    text = (
        "What does the author say about the role of intention in acts "
        "of worship and daily habits?"
    )
    assert len(text) > 60
    title = s.set_title_from_message(c["id"], text)
    assert s.get(c["id"])["title"] == title
    assert len(title) <= 60
    assert title.endswith("…")
    body = title[:-1]
    assert text.startswith(body)
    assert text[len(body)] == " "  # cut exactly at a word boundary


def test_auto_title_truncates_on_word_boundary_arabic():
    s = ConversationStore(_tmp_db())
    c = s.create()
    text = (
        "ما هي الشروط الواجب توافرها في الإمام عند صلاة الجماعة "
        "وما حكم من صلى خلف من لا يحفظ الفاتحة عند جمهور الفقهاء"
    )
    assert len(text) > 60
    title = s.set_title_from_message(c["id"], text)
    assert s.get(c["id"])["title"] == title
    assert len(title) <= 60
    assert title.endswith("…")
    body = title[:-1]
    assert text.startswith(body)
    assert text[len(body)] == " "  # Arabic words split on spaces too


def test_auto_title_short_message_kept_verbatim():
    s = ConversationStore(_tmp_db())
    c = s.create()
    assert s.set_title_from_message(c["id"], "  Short  question \n") == "Short question"
    title = s.get(c["id"])["title"]
    assert title == "Short question"
    assert not title.endswith("…")


def test_auto_title_noop_when_title_exists():
    s = ConversationStore(_tmp_db())
    c = s.create(title="Existing")
    assert s.set_title_from_message(c["id"], "something else entirely") is None
    assert s.get(c["id"])["title"] == "Existing"


def test_unknown_conversation_behaviors():
    s = ConversationStore(_tmp_db())
    assert s.get("nope") is None
    assert s.rename("nope", "t") is False
    assert s.delete("nope") is False
    assert s.set_title_from_message("nope", "hello") is None
    try:
        s.append_message("nope", "user", "hi")
    except KeyError:
        pass
    else:
        raise AssertionError("append_message must raise KeyError for unknown conversation")


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
