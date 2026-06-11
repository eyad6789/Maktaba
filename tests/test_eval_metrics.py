"""Unit tests for the eval CLI's pure helpers: ranking metrics, settings
overrides, ``--set`` parsing, and the expansion-cache memoizer.

CI-safe: pydantic models and pure functions only — no qdrant/embedder/LLM deps
(scripts.eval keeps its heavy imports lazy).
"""

from __future__ import annotations

import json
import math

import pytest

from config import settings
from core.models import SearchResult
from scripts.eval import (
    compute_metrics,
    memoize_expansion,
    override_settings,
    parse_set_overrides,
    rank_of_chunk,
    rank_of_first_match,
)


def _result(chunk_id: str, *, book="b1", page_start=1, page_end=1, index=0,
            rerank=None) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_id,
        score=0.0,
        text=f"text of {chunk_id}",
        book_id=book,
        title="Book",
        author=None,
        page_start=page_start,
        page_end=page_end,
        chunk_index=index,
        lang="en",
        rerank_score=rerank,
    )


# -- compute_metrics ----------------------------------------------------------


def test_compute_metrics_empty_list_is_all_zero() -> None:
    assert compute_metrics([], k=8) == {
        "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "hits": 0, "total": 0,
    }


def test_compute_metrics_all_misses() -> None:
    m = compute_metrics([None, None, None], k=8)
    assert m["recall"] == 0.0 and m["mrr"] == 0.0 and m["ndcg"] == 0.0
    assert m["hits"] == 0 and m["total"] == 3


def test_compute_metrics_rank_one_is_perfect() -> None:
    m = compute_metrics([1], k=8)
    assert m["recall"] == m["mrr"] == m["ndcg"] == 1.0
    assert m["hits"] == 1 and m["total"] == 1


def test_compute_metrics_rank_three_with_k_eight() -> None:
    m = compute_metrics([3], k=8)
    assert m["recall"] == 1.0
    assert abs(m["mrr"] - 1 / 3) < 1e-9
    assert abs(m["ndcg"] - 1 / math.log2(4)) < 1e-9


def test_compute_metrics_truncates_all_metrics_at_k() -> None:
    # Truncated MRR@k: a gold ranked beyond k contributes 0 to MRR as well —
    # NOT 1/rank — so every metric reflects only the top-k list the answer
    # model sees (documented in the compute_metrics docstring).
    assert compute_metrics([9], k=8) == {
        "recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "hits": 0, "total": 1,
    }


def test_compute_metrics_averages_mixed_ranks() -> None:
    m = compute_metrics([1, None, 4], k=8)
    assert abs(m["recall"] - 2 / 3) < 1e-9
    assert abs(m["mrr"] - (1 + 1 / 4) / 3) < 1e-9
    assert abs(m["ndcg"] - (1 + 1 / math.log2(5)) / 3) < 1e-9
    assert m["hits"] == 2 and m["total"] == 3


# -- rank_of_first_match --------------------------------------------------------


def test_rank_of_first_match_book_only() -> None:
    results = [_result("a", book="other"), _result("b", book="gold")]
    assert rank_of_first_match(results, "gold", None) == 2


def test_rank_of_first_match_page_span_inclusive() -> None:
    results = [
        _result("a", book="gold", page_start=1, page_end=3),
        _result("b", book="gold", page_start=4, page_end=6),
    ]
    assert rank_of_first_match(results, "gold", 5) == 2   # inside the span
    assert rank_of_first_match(results, "gold", 3) == 1   # boundary is inclusive
    assert rank_of_first_match(results, "gold", 4) == 2   # left boundary too


def test_rank_of_first_match_page_outside_all_spans_is_none() -> None:
    results = [_result("a", book="gold", page_start=1, page_end=3)]
    assert rank_of_first_match(results, "gold", 9) is None


def test_rank_of_first_match_absent_book_is_none() -> None:
    assert rank_of_first_match([_result("a", book="x")], "gold", None) is None


def test_rank_of_first_match_no_expectations_is_smoke_test() -> None:
    # Mirrors is_hit: with no gold at all, the first result "matches".
    assert rank_of_first_match([_result("a")], None, None) == 1
    assert rank_of_first_match([], None, None) is None


# -- rank_of_chunk ---------------------------------------------------------------


def test_rank_of_chunk_present() -> None:
    results = [_result("a"), _result("b"), _result("c")]
    assert rank_of_chunk(results, "b") == 2


def test_rank_of_chunk_absent_or_unset_is_none() -> None:
    results = [_result("a")]
    assert rank_of_chunk(results, "zzz") is None
    assert rank_of_chunk(results, None) is None
    assert rank_of_chunk([], "a") is None


# -- override_settings -------------------------------------------------------------


def test_override_settings_applies_and_restores() -> None:
    before = settings.rerank_min_score
    with override_settings(rerank_min_score=0.5):
        assert settings.rerank_min_score == 0.5
    assert settings.rerank_min_score == before


def test_override_settings_restores_after_exception() -> None:
    before = settings.enable_multi_query
    with pytest.raises(RuntimeError, match="boom"):
        with override_settings(enable_multi_query=not before):
            assert settings.enable_multi_query is (not before)
            raise RuntimeError("boom")
    assert settings.enable_multi_query is before


def test_override_settings_unknown_attribute_raises_attributeerror() -> None:
    # Pydantic itself raises ValueError ('"Settings" object has no field ...')
    # on unknown-field assignment (Settings has extra="ignore", which only
    # affects __init__); override_settings pre-checks with hasattr and raises
    # the conventional AttributeError before anything is mutated.
    with pytest.raises(AttributeError, match="no_such_knob"):
        with override_settings(no_such_knob=1):
            pass  # pragma: no cover - never entered
    assert not hasattr(settings, "no_such_knob")


def test_override_settings_multiple_knobs_all_restored() -> None:
    before = (settings.context_window_chunks, settings.rerank_min_score)
    with override_settings(context_window_chunks=0, rerank_min_score=0.25):
        assert settings.context_window_chunks == 0
        assert settings.rerank_min_score == 0.25
    assert (settings.context_window_chunks, settings.rerank_min_score) == before


# -- parse_set_overrides -------------------------------------------------------------


def test_parse_set_overrides_coerces_by_attribute_type() -> None:
    out = parse_set_overrides([
        "rerank_min_score=0.25",      # float attr
        "context_window_chunks=2",    # int attr
        "enable_multi_query=false",   # bool attr
        "openai_model=test-model",    # str attr
    ])
    assert out == {
        "rerank_min_score": 0.25,
        "context_window_chunks": 2,
        "enable_multi_query": False,
        "openai_model": "test-model",
    }
    assert isinstance(out["rerank_min_score"], float)
    assert isinstance(out["context_window_chunks"], int)
    assert out["enable_multi_query"] is False


def test_parse_set_overrides_bool_spellings() -> None:
    assert parse_set_overrides(["enable_multi_query=TRUE"])["enable_multi_query"] is True
    assert parse_set_overrides(["enable_multi_query=1"])["enable_multi_query"] is True
    assert parse_set_overrides(["enable_multi_query=off"])["enable_multi_query"] is False


def test_parse_set_overrides_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="no_such_knob"):
        parse_set_overrides(["no_such_knob=1"])


def test_parse_set_overrides_rejects_malformed_pair() -> None:
    with pytest.raises(ValueError, match="name=value"):
        parse_set_overrides(["rerank_min_score"])


def test_parse_set_overrides_rejects_uncoercible_values() -> None:
    with pytest.raises(ValueError, match="int"):
        parse_set_overrides(["context_window_chunks=two"])
    with pytest.raises(ValueError, match="float"):
        parse_set_overrides(["rerank_min_score=high"])
    with pytest.raises(ValueError, match="boolean"):
        parse_set_overrides(["enable_multi_query=maybe"])


# -- memoize_expansion --------------------------------------------------------------


def _cache_key(question: str) -> str:
    """The memoizer's key: variant count + question (see memoize_expansion)."""
    from config import settings

    return f"v{settings.multi_query_variants}|{question}"


def test_memoize_expansion_calls_underlying_once_per_question() -> None:
    calls: list[str] = []

    def fake_expand(question: str, **_: object) -> list[str]:
        calls.append(question)
        return [question + " variant"]

    cache: dict[str, list[str]] = {}
    wrapped = memoize_expansion(fake_expand, cache)
    assert wrapped("q1") == ["q1 variant"]
    assert wrapped("q1") == ["q1 variant"]   # served from the cache
    assert wrapped("q2") == ["q2 variant"]
    assert calls == ["q1", "q2"]


def test_memoize_expansion_cache_round_trips_through_json() -> None:
    cache: dict[str, list[str]] = {}
    memoize_expansion(lambda q, **_: ["v-" + q], cache)("سؤال بالعربية")
    restored = json.loads(json.dumps(cache, ensure_ascii=False))
    assert restored == {_cache_key("سؤال بالعربية"): ["v-سؤال بالعربية"]}

    def explode(question: str, **_: object) -> list[str]:
        raise AssertionError("the restored cache should have answered")

    assert memoize_expansion(explode, restored)("سؤال بالعربية") == ["v-سؤال بالعربية"]


def test_memoize_expansion_returns_copies() -> None:
    cache: dict[str, list[str]] = {}
    wrapped = memoize_expansion(lambda q, **_: ["a"], cache)
    wrapped("q").append("mutated")
    assert wrapped("q") == ["a"]
    assert cache[_cache_key("q")] == ["a"]


def test_memoize_expansion_never_caches_empty_results() -> None:
    """expand_queries returns [] on LLM failure — a miss, not a cacheable fact."""
    calls: list[str] = []
    answers: list[list[str]] = [[], ["recovered variant"]]

    def flaky_expand(question: str, **_: object) -> list[str]:
        calls.append(question)
        return answers[len(calls) - 1]

    cache: dict[str, list[str]] = {}
    wrapped = memoize_expansion(flaky_expand, cache)
    assert wrapped("q") == []          # transient failure surfaces as empty...
    assert cache == {}                 # ...but is never persisted
    assert wrapped("q") == ["recovered variant"]   # retried on the next call
    assert cache == {_cache_key("q"): ["recovered variant"]}


def test_memoize_expansion_keys_include_variant_count() -> None:
    """--set multi_query_variants=N must not reuse another count's variants."""
    from scripts.eval import override_settings

    cache: dict[str, list[str]] = {}
    calls: list[str] = []

    def fake_expand(question: str, **_: object) -> list[str]:
        calls.append(question)
        return [f"variant {len(calls)}"]

    wrapped = memoize_expansion(fake_expand, cache)
    with override_settings(multi_query_variants=2):
        assert wrapped("q") == ["variant 1"]
    with override_settings(multi_query_variants=3):
        assert wrapped("q") == ["variant 2"]   # different count -> fresh expansion
        assert wrapped("q") == ["variant 2"]   # ...then cached under its own key
    assert len(calls) == 2


# -- model memoization wrappers ------------------------------------------------


class _CountingEmbedder:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed_query(self, text: str) -> str:
        self.calls.append(text)
        return f"emb({text})"


class _CountingReranker:
    """Scores by the trailing digit of the chunk_id (deterministic, fake)."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query, results, top_k=8):
        self.calls += 1
        for r in results:
            r.rerank_score = int(r.chunk_id[-1]) / 10
        return sorted(results, key=lambda r: r.rerank_score, reverse=True)[:top_k]


def test_memo_embedder_embeds_each_text_once() -> None:
    from scripts.eval import _MemoEmbedder

    inner = _CountingEmbedder()
    memo = _MemoEmbedder(inner)
    assert memo.embed_query("q") == "emb(q)"
    assert memo.embed_query("q") == "emb(q)"
    assert memo.embed_query("other") == "emb(other)"
    assert inner.calls == ["q", "other"]


def test_memo_reranker_reuses_scores_without_aliasing() -> None:
    from scripts.eval import _MemoReranker

    inner = _CountingReranker()
    memo = _MemoReranker(inner)

    first = [_result("c1"), _result("c3"), _result("c2")]
    out1 = memo.rerank("q", first, top_k=2)
    assert inner.calls == 1
    assert [r.chunk_id for r in out1] == ["c3", "c2"]

    # Fresh objects, same (query, chunk) pairs -> served from the score cache.
    second = [_result("c2"), _result("c1"), _result("c3")]
    out2 = memo.rerank("q", second, top_k=3)
    assert inner.calls == 1
    assert [r.chunk_id for r in out2] == ["c3", "c2", "c1"]
    assert [r.rerank_score for r in out2] == [0.3, 0.2, 0.1]
    # Identity (not ==): pydantic models compare by value, aliasing is the point.
    assert all(any(r is s for s in second) for r in out2)   # the fresh objects
    assert not any(any(r is f for f in first) for r in out2)  # never the old ones

    # A new chunk in the mix forces one more real call; cache then covers all.
    third = [_result("c1"), _result("c4")]
    memo.rerank("q", third, top_k=2)
    assert inner.calls == 2

    # Different query -> different cache keys -> real call.
    memo.rerank("q2", [_result("c1")], top_k=1)
    assert inner.calls == 3


def test_memo_embedder_persists_and_reloads(tmp_path) -> None:
    from core.models import Embedding, SparseVector
    from scripts.eval import _MemoEmbedder

    class EmbeddingEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed_query(self, text: str) -> Embedding:
            self.calls += 1
            return Embedding(
                dense=[0.5, 0.25], sparse=SparseVector(indices=[3], values=[1.0]),
            )

    path = tmp_path / "embed_cache.json"
    inner = EmbeddingEmbedder()
    memo = _MemoEmbedder(inner, persist_path=path)
    memo.embed_query("سؤال بالعربية")
    memo.save()
    assert path.is_file()

    fresh_inner = EmbeddingEmbedder()
    reloaded = _MemoEmbedder(fresh_inner, persist_path=path)
    emb = reloaded.embed_query("سؤال بالعربية")
    assert fresh_inner.calls == 0                  # served from disk
    assert emb.dense == [0.5, 0.25]
    assert emb.sparse.indices == [3]

    reloaded.save()                                 # nothing new -> no rewrite
    assert inner.calls == 1


def test_memo_reranker_persists_and_reloads(tmp_path) -> None:
    from scripts.eval import _MemoReranker

    path = tmp_path / "rerank_cache.json"
    inner = _CountingReranker()
    memo = _MemoReranker(inner, persist_path=path)
    memo.rerank("q", [_result("c1"), _result("c2")], top_k=2)
    memo.save()
    assert path.is_file()

    fresh_inner = _CountingReranker()
    reloaded = _MemoReranker(fresh_inner, persist_path=path)
    out = reloaded.rerank("q", [_result("c2"), _result("c1")], top_k=2)
    assert fresh_inner.calls == 0                  # served from disk
    assert [r.chunk_id for r in out] == ["c2", "c1"]
    assert [r.rerank_score for r in out] == [0.2, 0.1]


def test_memo_wrappers_ignore_corrupt_cache_files(tmp_path) -> None:
    from scripts.eval import _MemoEmbedder, _MemoReranker

    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    embedder = _MemoEmbedder(_CountingEmbedder(), persist_path=bad)
    assert embedder.embed_query("q") == "emb(q)"   # falls back to live model
    reranker = _MemoReranker(_CountingReranker(), persist_path=bad)
    assert reranker.rerank("q", [_result("c1")], top_k=1)
