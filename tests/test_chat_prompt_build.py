"""Pure prompt/citation helpers shared by chat and the streaming endpoint.

`build_chat_messages` must graft the grounded prompt onto ONLY the last user
turn (history passes through verbatim), and `extract_citations` must map [n]
markers back to sources and decide groundedness. Pure string/model logic — no
engine, network, or SDKs.
"""

from __future__ import annotations

from core.models import SearchResult
from llm.answer import extract_citations
from llm.chat import build_chat_messages
from retrieval.route import Route


def _result(i: int, title: str = "Book One") -> SearchResult:
    return SearchResult(
        chunk_id=f"c{i}", score=1.0, text=f"passage {i}", book_id="b1",
        title=title, author="Author", page_start=i, page_end=i + 1, chunk_index=i,
    )


def _messages() -> list[dict]:
    return [
        {"role": "user", "content": "What is tawhid?"},
        {"role": "assistant", "content": "Tawhid is ... [1]"},
        {"role": "user", "content": "And its categories?"},
    ]


# --- build_chat_messages -----------------------------------------------------


def test_replaces_only_last_user_turn_with_grounded_prompt():
    messages = _messages()
    results = [_result(1), _result(2)]
    system, api_messages = build_chat_messages(messages, results, Route.LOCAL)

    assert len(api_messages) == 3
    # Earlier turns pass through verbatim, order preserved.
    assert api_messages[0] == {"role": "user", "content": "What is tawhid?"}
    assert api_messages[1] == {"role": "assistant", "content": "Tawhid is ... [1]"}
    # The last user turn becomes the full grounded prompt: context + question.
    last = api_messages[2]
    assert last["role"] == "user"
    assert "Context (numbered sources):" in last["content"]
    assert "passage 1" in last["content"] and "passage 2" in last["content"]
    assert "Question: And its categories?" in last["content"]


def test_input_messages_are_not_mutated():
    messages = _messages()
    build_chat_messages(messages, [_result(1)], Route.LOCAL)
    assert messages[-1]["content"] == "And its categories?"


def test_system_prompt_follows_route_and_keeps_multiturn_rule():
    local_system, _ = build_chat_messages(_messages(), [_result(1)], Route.LOCAL)
    global_system, _ = build_chat_messages(_messages(), [_result(1)], Route.GLOBAL)

    assert "multi-turn conversation" in local_system
    assert "multi-turn conversation" in global_system
    # GLOBAL gets the whole-book synthesis addendum; LOCAL stays tight.
    assert "WHOLE-BOOK / THEMATIC" in global_system
    assert "WHOLE-BOOK / THEMATIC" not in local_system


def test_single_turn_conversation_still_grounds_the_only_user_message():
    messages = [{"role": "user", "content": "Define fiqh"}]
    _, api_messages = build_chat_messages(messages, [_result(1)], Route.LOCAL)
    assert len(api_messages) == 1
    assert "Question: Define fiqh" in api_messages[0]["content"]


# --- extract_citations ---------------------------------------------------------


def test_extract_citations_maps_markers_to_sources():
    results = [_result(1, "Book One"), _result(2, "Book Two")]
    citations, grounded = extract_citations(
        "Tawhid has three categories [2], as explained [1][2].", results
    )
    # First-seen order, de-duplicated; [n] maps to results[n-1].
    assert [c.title for c in citations] == ["Book Two", "Book One"]
    assert citations[0].page_start == 2
    assert grounded is True


def test_extract_citations_ignores_out_of_range_markers():
    citations, grounded = extract_citations("See [1] and also [9].", [_result(1)])
    assert len(citations) == 1
    assert grounded is True


def test_not_found_answer_is_ungrounded():
    citations, grounded = extract_citations(
        "This information was not found in the books.", [_result(1)]
    )
    assert citations == []
    assert grounded is False


def test_citations_outweigh_not_found_phrasing():
    # A real citation wins even if the wording also hedges with "not found".
    _, grounded = extract_citations(
        "The detail was not found in the books, but the theme appears [1].",
        [_result(1)],
    )
    assert grounded is True


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
