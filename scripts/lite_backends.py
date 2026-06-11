"""Lightweight CPU stand-ins for the GPU model stack — DEV / DEMO ONLY.

These let the whole app run on a laptop without a GPU or the multi-GB BGE-M3 /
reranker downloads. Retrieval quality is approximate (lexical hashing, not
semantic), so this is for local demos and development — NOT production.

* :class:`HashingEmbedder` — a deterministic hashing vectorizer with the same
  interface as :class:`ingest.embed.BGEM3Embedder` (dense bag-of-words hashed
  into ``settings.embedding_dim`` buckets + a lexical sparse vector).
* :class:`OverlapReranker` — token-overlap scorer with the same interface as
  :class:`retrieval.rerank.Reranker`.
"""

from __future__ import annotations

import hashlib
import math
import re

from config import settings
from core.models import Embedding, SearchResult, SparseVector

_TOKEN_RE = re.compile(r"[0-9A-Za-z؀-ۿ]+")
_SPARSE_VOCAB = 100_003


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _bucket(token: str, mod: int) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % mod


class HashingEmbedder:
    """Deterministic lexical embedder (dev stand-in for BGE-M3)."""

    def __init__(self, dim: int | None = None, *_, **__) -> None:
        self.dim = dim or settings.embedding_dim

    def _embed_one(self, text: str) -> Embedding:
        toks = _tokens(text)
        dense = [0.0] * self.dim
        sparse: dict[int, float] = {}
        for t in toks:
            dense[_bucket(t, self.dim)] += 1.0
            si = _bucket("s:" + t, _SPARSE_VOCAB)
            sparse[si] = sparse.get(si, 0.0) + 1.0
        norm = math.sqrt(sum(x * x for x in dense)) or 1.0
        dense = [x / norm for x in dense]
        if not sparse:
            sparse = {0: 1e-6}
        return Embedding(
            dense=dense,
            sparse=SparseVector(indices=list(sparse.keys()), values=list(sparse.values())),
        )

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        return [self._embed_one(t) for t in texts]

    def embed_queries(self, texts: list[str]) -> list[Embedding]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> Embedding:
        return self._embed_one(text)


class OverlapReranker:
    """Token-overlap reranker (dev stand-in for bge-reranker-v2-m3)."""

    def __init__(self, *_, **__) -> None:
        pass

    def rerank(self, query: str, results: list[SearchResult], top_k: int = settings.rerank_top_k):
        q = set(_tokens(query))
        for r in results:
            overlap = len(q & set(_tokens(r.text)))
            r.rerank_score = overlap / (len(q) + 1.0)
        ranked = sorted(results, key=lambda r: (r.rerank_score or 0.0), reverse=True)
        limit = top_k if top_k and top_k > 0 else len(ranked)
        return ranked[:limit]


class SentenceTransformerEmbedder:
    """REAL multilingual semantic embedder for the demo (CPU-friendly).

    Dense vectors come from a sentence-transformers model (paraphrase-
    multilingual-MiniLM, 384-d, Arabic + English); a lexical token-hash sparse
    vector is added so hybrid (dense + sparse) search still works. Far better
    retrieval than :class:`HashingEmbedder`. Set ``EMBEDDING_DIM`` to match the
    model's dimension (384 for MiniLM).
    """

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", *_, **__) -> None:
        self.model_name = model_name
        self._model = None

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def _sparse(self, text: str) -> SparseVector:
        sparse: dict[int, float] = {}
        for t in _tokens(text):
            si = _bucket("s:" + t, _SPARSE_VOCAB)
            sparse[si] = sparse.get(si, 0.0) + 1.0
        if not sparse:
            sparse = {0: 1e-6}
        return SparseVector(indices=list(sparse.keys()), values=list(sparse.values()))

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        if not texts:
            return []
        model = self._load()
        vecs = model.encode(list(texts), normalize_embeddings=True, batch_size=16)
        return [
            Embedding(dense=[float(x) for x in v], sparse=self._sparse(t))
            for t, v in zip(texts, vecs)
        ]

    def embed_queries(self, texts: list[str]) -> list[Embedding]:
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> Embedding:
        return self.embed_documents([text])[0]


class SemanticReranker:
    """Rerank candidates by cosine similarity of dense embeddings (reuses an
    embedder). Keeps reranking semantic and consistent with dense retrieval."""

    def __init__(self, embedder, *_, **__) -> None:
        self.embedder = embedder

    def rerank(self, query: str, results: list[SearchResult], top_k: int = settings.rerank_top_k):
        if not results:
            return []
        import numpy as np

        q = np.asarray(self.embedder.embed_query(query).dense, dtype="float32")
        qn = float(np.linalg.norm(q)) or 1.0
        doc_vecs = self.embedder.embed_documents([r.text for r in results])
        for r, emb in zip(results, doc_vecs):
            d = np.asarray(emb.dense, dtype="float32")
            r.rerank_score = float(q @ d) / (qn * (float(np.linalg.norm(d)) or 1.0))
        ranked = sorted(results, key=lambda r: (r.rerank_score or 0.0), reverse=True)
        limit = top_k if top_k and top_k > 0 else len(ranked)
        return ranked[:limit]
