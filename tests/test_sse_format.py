"""Unit tests for the SSE frame formatter (core/sse.py)."""

from __future__ import annotations

import json

from core.sse import format_sse


def test_basic_frame_structure() -> None:
    frame = format_sse("delta", {"text": "hello"})
    assert frame.startswith("event: delta\n")
    assert frame.endswith("\n\n")
    lines = frame.strip().split("\n")
    assert lines[0] == "event: delta"
    assert lines[1].startswith("data: ")


def test_data_is_valid_json() -> None:
    frame = format_sse("meta", {"conversation_id": "abc", "sources": [1, 2]})
    payload = frame.strip().split("\n", 1)[1].removeprefix("data: ")
    assert json.loads(payload) == {"conversation_id": "abc", "sources": [1, 2]}


def test_arabic_not_escaped() -> None:
    frame = format_sse("delta", {"text": "ما الفكرة الرئيسية؟"})
    assert "ما الفكرة الرئيسية؟" in frame  # ensure_ascii=False keeps Arabic readable


def test_payload_stays_single_line() -> None:
    # JSON encodes newlines as \n escapes, so one data: line always suffices.
    frame = format_sse("delta", {"text": "line1\nline2"})
    assert frame.count("\ndata: ") == 1
    body = frame.strip().split("\n", 1)[1]
    assert "\n" not in body
