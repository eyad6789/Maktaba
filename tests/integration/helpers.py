"""Test doubles + PDF generators for the end-to-end integration run.

These let us exercise the *real* pipeline / Qdrant / API / queue code paths
without a GPU or network: a deterministic hashing embedder stands in for BGE-M3,
a token-overlap reranker stands in for the cross-encoder, a canned OCR backend
stands in for Qwen/Surya, and a fake Anthropic client stands in for Claude.
"""

from __future__ import annotations

import hashlib
import io
import math
import re

from config import settings
from core.models import Embedding, SearchResult, SparseVector

# Tokenizer covering Latin + Arabic word characters.
_TOKEN_RE = re.compile(r"[0-9A-Za-z؀-ۿ]+")
_SPARSE_VOCAB = 100_003  # prime-ish bucket count for the fake sparse space


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _bucket(token: str, mod: int) -> int:
    return int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % mod


class FakeEmbedder:
    """Deterministic hashing vectorizer with the BGEM3Embedder interface.

    Dense: bag-of-words hashed into `dim` buckets, L2-normalized (so cosine
    similarity reflects shared tokens). Sparse: token-id hashing with TF weights.
    Same-language text sharing words is retrieved together — enough to verify
    real hybrid search + RRF without the actual model.
    """

    def __init__(self, dim: int | None = None) -> None:
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
        if not sparse:  # never emit a truly empty sparse vector
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


class FakeReranker:
    """Token-overlap reranker with the Reranker interface."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def rerank(self, query: str, results: list[SearchResult], top_k: int = settings.rerank_top_k):
        q = set(_tokens(query))
        for r in results:
            overlap = len(q & set(_tokens(r.text)))
            r.rerank_score = overlap / (len(q) + 1.0)
        ranked = sorted(results, key=lambda r: (r.rerank_score or 0.0), reverse=True)
        limit = top_k if top_k and top_k > 0 else len(ranked)
        return ranked[:limit]


class FakeOCR:
    """OCR backend returning a fixed transcription, ignoring the image."""

    name = "fake"

    def __init__(self, text: str) -> None:
        self.text = text

    def ocr_image(self, _image) -> str:  # noqa: ANN001 - duck-typed PIL image
        return self.text


# --- fake LLM engine --------------------------------------------------------


def install_fake_llm(answer_text: str) -> None:
    """Monkeypatch the LLM engine to return a canned completion (no model/API).

    Patches ``llm.engine.complete`` (still used by ``condense_query`` and the
    ingest summarizer) AND ``llm.engine.generate`` (the provider-aware call
    ``llm.answer``/``llm.chat`` now make), so generation is deterministic and
    offline in tests regardless of which entrypoint the production code uses.
    """
    import llm.engine as engine_mod

    engine_mod.complete = lambda system, messages, **kwargs: answer_text  # type: ignore[assignment]
    engine_mod.generate = lambda system, messages, **kwargs: engine_mod.GenResult(  # type: ignore[assignment]
        text=answer_text, provider="gemini", model="fake-model"
    )


def install_fake_llm_stream(
    answer_text: str,
    provider_id: str = "gemini",
    model: str = "fake-model",
    n_chunks: int = 3,
) -> None:
    """Monkeypatch ``llm.engine.stream`` to stream a canned answer (no API).

    The fake yields exactly what the real engine does: one
    ``("provider", {"provider", "model"})`` event, then ``answer_text`` split
    into (up to) ``n_chunks`` ``("delta", piece)`` tuples — so the SSE endpoint
    is exercised end to end with deterministic output.
    """
    import llm.engine as engine_mod

    def fake_stream(system, messages, **kwargs):  # noqa: ANN001, ANN003 - duck-typed
        yield ("provider", {"provider": provider_id, "model": model})
        size = max(1, -(-len(answer_text) // max(1, n_chunks)))  # ceil division
        for i in range(0, len(answer_text), size):
            yield ("delta", answer_text[i : i + size])

    engine_mod.stream = fake_stream  # type: ignore[assignment]


def install_fake_llm_stream_rate_limited(provider_id: str = "gemini") -> None:
    """Monkeypatch ``llm.engine.stream`` to hit a rate limit before any token.

    Mirrors a pinned provider 429ing: the generator raises
    ``ProviderError(reason="rate_limit")`` before its first yield, which the
    streaming endpoint must surface as a single terminal ``error`` SSE event
    (and must NOT persist an assistant turn).
    """
    import llm.engine as engine_mod
    from llm.errors import ProviderError

    def fake_stream(system, messages, **kwargs):  # noqa: ANN001, ANN003 - duck-typed
        raise ProviderError(provider_id, "rate_limit", "429 quota exceeded")
        yield  # pragma: no cover - unreachable; makes this a generator function

    engine_mod.stream = fake_stream  # type: ignore[assignment]


# --- PDF generators (real PyMuPDF) ------------------------------------------


def make_native_pdf(path, page_texts: list[str]) -> None:
    """Write a native-text PDF: one page per string (extractable via get_text)."""
    import fitz

    doc = fitz.open()
    for text in page_texts:
        page = doc.new_page()
        rect = fitz.Rect(40, 40, page.rect.width - 40, page.rect.height - 40)
        page.insert_textbox(rect, text, fontsize=10, fontname="helv")
    doc.save(str(path))
    doc.close()


def make_scanned_pdf(path, width: int = 1000, height: int = 1400) -> None:
    """Write an image-only PDF page (no text layer) -> classifies as SCANNED."""
    import fitz
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, width - 40, height - 40], outline="black", width=3)
    draw.text((80, 80), "[simulated scanned page]", fill="black")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    doc = fitz.open()
    page = doc.new_page(width=width, height=height)
    page.insert_image(fitz.Rect(0, 0, width, height), stream=buf.read())
    doc.save(str(path))
    doc.close()
