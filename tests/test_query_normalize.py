"""Query-time normalization (the retrieval side of ``ingest.normalize``).

``retrieve()`` must hand the embedder, the expansion layer, and the reranker
the SAME Arabic-normalized question that the indexed text received — otherwise
the sparse/lexical channel misses hamza/maqsura/tatweel spelling variants and
the cross-encoder scores a raw query against normalized passages.

CI-safe: fake store/embedder/reranker, no heavy deps.
"""

from __future__ import annotations

from config import settings
from core.models import SearchResult
from ingest.normalize import normalize_text
from retrieval.pipeline import retrieve
from retrieval.route import Route


class RecordingEmbedder:
    """Records every text it is asked to embed; the embedding is unused."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.batches: list[list[str]] = []

    def embed_queries(self, texts: list[str]):
        self.batches.append(list(texts))
        self.texts.extend(texts)
        return [None] * len(texts)

    def embed_query(self, text: str):
        return self.embed_queries([text])[0]


class RecordingReranker:
    """Records the query it scores against; keeps input order."""

    def __init__(self) -> None:
        self.queries: list[str] = []

    def rerank(self, query, results, top_k=8):
        self.queries.append(query)
        for r in results:
            r.rerank_score = 1.0
        return list(results)[:top_k]


class StubStore:
    def hybrid_search(self, query, top_k=50, book_ids=None, levels=None):
        return [
            SearchResult(
                chunk_id="p1", score=0.0, text="t", book_id="b1", title="T",
                page_start=1, page_end=1, chunk_index=0,
            )
        ]


# Hamza-carrying alefs (أ / إ) plus a tatweel run — exactly the spelling
# variants normalize_arabic folds at index time.
AR_RAW = "أَيْنَ ذكرَ المؤلفُ الإِحسانَـــ والعدلَ؟"


def test_arabic_question_is_normalized_before_embedding(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_multi_query", False)
    embedder = RecordingEmbedder()
    retrieve(
        AR_RAW, embedder=embedder, store=StubStore(),
        reranker=RecordingReranker(), route=Route.LOCAL,
    )
    assert embedder.texts == [normalize_text(AR_RAW)]
    assert embedder.texts[0] != AR_RAW       # the folding actually happened
    assert "أ" not in embedder.texts[0]      # hamza-alef folded to bare alef
    assert "إ" not in embedder.texts[0]
    assert "ـ" not in embedder.texts[0]      # tatweel stripped


def test_reranker_sees_the_normalized_question(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_multi_query", False)
    reranker = RecordingReranker()
    retrieve(
        AR_RAW, embedder=RecordingEmbedder(), store=StubStore(),
        reranker=reranker, route=Route.LOCAL,
    )
    assert reranker.queries == [normalize_text(AR_RAW)]


def test_english_question_passes_through_unchanged(monkeypatch) -> None:
    monkeypatch.setattr(settings, "enable_multi_query", False)
    q = "Where does the author discuss justice and benevolence?"
    embedder = RecordingEmbedder()
    retrieve(
        q, embedder=embedder, store=StubStore(),
        reranker=RecordingReranker(), route=Route.LOCAL,
    )
    assert embedder.texts == [q]


def test_expansion_variants_are_normalized_too(monkeypatch) -> None:
    import retrieval.expand as expand_mod

    variant_raw = "إِحسانٌ وعدلـــٌ مرةً أُخرى"
    monkeypatch.setattr(settings, "enable_multi_query", True)
    monkeypatch.setattr(expand_mod, "expand_queries", lambda q, **kw: [variant_raw])

    embedder = RecordingEmbedder()
    retrieve(
        AR_RAW, embedder=embedder, store=StubStore(),
        reranker=RecordingReranker(), route=Route.LOCAL,
    )
    assert len(embedder.texts) == 2
    assert embedder.texts[1] == normalize_text(variant_raw)
    assert "إ" not in embedder.texts[1]
    assert "ـ" not in embedder.texts[1]


def test_question_and_variants_embed_in_one_batched_call(monkeypatch) -> None:
    """Efficiency contract: ONE embed_queries call for question + variants."""
    import retrieval.expand as expand_mod

    monkeypatch.setattr(settings, "enable_multi_query", True)
    monkeypatch.setattr(
        expand_mod, "expand_queries", lambda q, **kw: ["variant one", "variant two"]
    )
    embedder = RecordingEmbedder()
    retrieve(
        "Where does the author discuss justice?", embedder=embedder,
        store=StubStore(), reranker=RecordingReranker(), route=Route.LOCAL,
    )
    assert len(embedder.batches) == 1                 # single batched model call
    assert len(embedder.batches[0]) == 3              # question + 2 variants


def test_batched_embed_failure_falls_back_to_question_alone(monkeypatch) -> None:
    import retrieval.expand as expand_mod

    monkeypatch.setattr(settings, "enable_multi_query", True)
    monkeypatch.setattr(expand_mod, "expand_queries", lambda q, **kw: ["variant"])

    class FlakyEmbedder(RecordingEmbedder):
        def embed_queries(self, texts):
            if len(texts) > 1:
                raise RuntimeError("batch failed")
            return super().embed_queries(texts)

    embedder = FlakyEmbedder()
    results = retrieve(
        "Where does the author discuss justice?", embedder=embedder,
        store=StubStore(), reranker=RecordingReranker(), route=Route.LOCAL,
    )
    assert results                                    # retrieval still answered
    assert embedder.batches == [["Where does the author discuss justice?"]]
