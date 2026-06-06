"""Query routing heuristics (Phase 5).

The heuristic must classify thematic vs factual questions in both languages and
make NO model call on the hot path. The optional LLM tie-breaker runs only when
`router_use_llm` is set.
"""

from __future__ import annotations

import llm.engine as engine_mod
from config import settings
from retrieval.route import (
    Route,
    classify_route,
    coerce_route,
    heuristic_route,
)


def test_english_thematic_routes_global():
    for q in [
        "What is the main idea of this book?",
        "Summarize the chapter on justice",
        "What is the author's argument?",
        "what is this book about",
        "Compare the two views of liberty",
    ]:
        assert heuristic_route(q) == Route.GLOBAL, q


def test_arabic_thematic_routes_global():
    for q in [
        "ما هي الفكرة الرئيسية في الكتاب؟",
        "لخص الفصل الأول",
        "ما موضوع الكتاب؟",
        "قارن بين الرأيين",
    ]:
        assert heuristic_route(q) == Route.GLOBAL, q


def test_factual_routes_local():
    for q in ["How many chapters are there?", "كم عدد الصفحات؟", "Who is the author?"]:
        assert heuristic_route(q) == Route.LOCAL, q


def test_global_cue_wins_over_local_cue():
    # "how many" (LOCAL) + "main ideas" (GLOBAL) -> GLOBAL wins (checked first).
    assert heuristic_route("how many main ideas are there") == Route.GLOBAL


def test_ambiguous_returns_none():
    assert heuristic_route("tell me about justice and fairness") is None


def test_classify_defaults_to_local_without_engine_call():
    # With router_use_llm off, an ambiguous question must NOT touch the LLM.
    orig_complete = engine_mod.complete
    orig_flag = settings.router_use_llm
    settings.router_use_llm = False

    def boom(*a, **k):
        raise AssertionError("engine.complete must not be called on the hot path")

    engine_mod.complete = boom
    try:
        assert classify_route("an utterly ambiguous sentence") == Route.LOCAL
    finally:
        engine_mod.complete = orig_complete
        settings.router_use_llm = orig_flag


def test_llm_tiebreaker_used_only_when_enabled():
    orig_complete = engine_mod.complete
    orig_flag = settings.router_use_llm
    settings.router_use_llm = True
    engine_mod.complete = lambda system, messages, **kw: "GLOBAL"
    try:
        assert classify_route("an utterly ambiguous sentence") == Route.GLOBAL
    finally:
        engine_mod.complete = orig_complete
        settings.router_use_llm = orig_flag


def test_coerce_route_parses_overrides():
    assert coerce_route("global") == Route.GLOBAL
    assert coerce_route("LOCAL") == Route.LOCAL
    assert coerce_route(Route.GLOBAL) == Route.GLOBAL
    assert coerce_route(None) is None
    assert coerce_route("") is None
    assert coerce_route("nonsense") is None


if __name__ == "__main__":
    from tests._runner import main

    main(globals())
