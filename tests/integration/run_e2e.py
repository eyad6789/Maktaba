"""End-to-end integration run against REAL Qdrant + Redis.

Exercises the actual pipeline / retrieval / answer / API / queue code with
deterministic test doubles for the GPU/cloud pieces (embedder, reranker, OCR,
Claude). Requires:

  * Qdrant reachable at QDRANT_URL (default http://localhost:6333)
  * Redis reachable at REDIS_URL  (set below to redis://localhost:6380/0)

Run:  .venv/bin/python -m tests.integration.run_e2e
"""

from __future__ import annotations

import os
import tempfile
import traceback
from pathlib import Path

# --- environment MUST be set before importing config/project modules ---------
_TMP = Path(tempfile.gettempdir()) / "rag_e2e"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("QDRANT_COLLECTION", "rag_e2e_test")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("RERANKER_DEVICE", "cpu")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key")
os.environ.setdefault("OCR_BACKEND", "tesseract")
os.environ.setdefault("REDIS_URL", "redis://localhost:6380/0")
os.environ.setdefault("REGISTRY_DB", str(_TMP / "registry.db"))
os.environ.setdefault("LOG_LEVEL", "WARNING")

from config import settings  # noqa: E402
from core.models import BookMeta, PageContent, PageKind  # noqa: E402
from ingest.chunk import chunk_pages  # noqa: E402
from ingest.classify import classify_pdf  # noqa: E402
from ingest.pipeline import ingest_book  # noqa: E402
from ingest.registry import Registry  # noqa: E402
from llm.answer import answer_question  # noqa: E402
from retrieval.search import QdrantStore  # noqa: E402
from tests.integration.helpers import (  # noqa: E402
    FakeEmbedder,
    FakeOCR,
    FakeReranker,
    install_fake_llm,
    make_native_pdf,
    make_scanned_pdf,
)

JUSTICE = (
    "Justice is the concept of fairness and law. A just society treats every "
    "citizen equally under the law. Fairness, equality, and the rule of law are "
    "the foundations of justice in political philosophy. Judges apply the law to "
    "ensure justice and fairness for all people in a just legal system."
)
ASTRONOMY = (
    "Astronomy studies stars, planets, and galaxies. The night sky contains "
    "billions of stars. Telescopes observe distant galaxies and the planets "
    "orbiting other stars. Cosmology explores the origin of the universe and the "
    "expansion of space among the stars and planets."
)
OCR_TEXT = (
    "العدالة القانون المجتمع الحوكمة المساواة. "
    "This simulated scanned page discusses governance and society."
)

_PDF_JUSTICE = _TMP / "justice.pdf"
_PDF_ASTRO = _TMP / "astronomy.pdf"
_PDF_SCANNED = _TMP / "scanned.pdf"

_embedder = FakeEmbedder()
_reranker = FakeReranker()
_state: dict[str, str] = {}  # book ids captured across tests


def _store() -> QdrantStore:
    return QdrantStore()  # uses test collection from env


def setup() -> None:
    """Reset Qdrant test collection + registry, regenerate sample PDFs."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    reg = Path(settings.registry_db)
    if reg.exists():
        reg.unlink()

    make_native_pdf(_PDF_JUSTICE, [JUSTICE])
    make_native_pdf(_PDF_ASTRO, [ASTRONOMY])
    make_scanned_pdf(_PDF_SCANNED)


# --- tests ------------------------------------------------------------------


def test_native_pdf_extraction():
    """Real PyMuPDF: native pages classify as NATIVE and text extracts + chunks."""
    kinds = classify_pdf(_PDF_JUSTICE)
    assert kinds == [PageKind.NATIVE], f"expected [NATIVE], got {kinds}"

    import fitz

    doc = fitz.open(str(_PDF_JUSTICE))
    try:
        text = doc.load_page(0).get_text("text")
    finally:
        doc.close()
    assert "Justice" in text and "law" in text, "extracted text missing expected words"

    pages = [PageContent(page_number=1, text=text, kind=PageKind.NATIVE, lang="en")]
    book = BookMeta(book_id="t", title="t", source_path="t", file_hash="t", num_pages=1)
    chunks = chunk_pages(pages, book, target_tokens=40, overlap_tokens=8, min_tokens=5)
    assert len(chunks) >= 1 and all(c.text.strip() for c in chunks)


def test_ingest_native_and_query():
    """Ingest two native PDFs into REAL Qdrant, then verify hybrid retrieval."""
    store = _store()
    reg = Registry()

    s1 = ingest_book(_PDF_JUSTICE, store=store, embedder=_embedder, registry=reg, title="Justice")
    s2 = ingest_book(_PDF_ASTRO, store=store, embedder=_embedder, registry=reg, title="Astronomy")
    assert s1.status == "completed" and s1.num_chunks >= 1, s1
    assert s2.status == "completed" and s2.num_chunks >= 1, s2
    assert s1.native_pages == 1 and s1.scanned_pages == 0, s1
    _state["justice_id"] = s1.book_id
    _state["astro_id"] = s2.book_id

    q = "justice fairness and the rule of law"
    cands = store.hybrid_search(_embedder.embed_query(q), top_k=settings.search_top_k)
    assert cands, "hybrid_search returned no candidates"
    ids = {c.book_id for c in cands}
    assert s1.book_id in ids, "justice book not retrieved"

    top = _reranker.rerank(q, cands, top_k=5)
    assert top[0].book_id == s1.book_id, (
        f"expected justice book on top, got {top[0].title!r}"
    )

    # Real answer assembly + citation parsing with a canned Claude reply.
    install_fake_llm("Justice is fairness under the law [1].")
    ans = answer_question(q, top, model=None)
    assert ans.grounded is True
    assert ans.citations and ans.citations[0].book_id == top[0].book_id
    assert ans.sources and ans.model


def test_ingest_scanned_with_ocr():
    """Image-only PDF routes to OCR; OCR'd Arabic text becomes retrievable."""
    store = _store()
    reg = Registry()

    stats = ingest_book(
        _PDF_SCANNED,
        store=store,
        embedder=_embedder,
        registry=reg,
        ocr=FakeOCR(OCR_TEXT),
        title="Scanned",
    )
    assert stats.status == "completed", stats
    assert stats.scanned_pages >= 1 and stats.native_pages == 0, stats
    assert stats.num_chunks >= 1, stats
    _state["scanned_id"] = stats.book_id

    q = "العدالة القانون"
    cands = store.hybrid_search(_embedder.embed_query(q), top_k=settings.search_top_k)
    top = _reranker.rerank(q, cands, top_k=5)
    assert top and top[0].book_id == stats.book_id, "Arabic OCR text not retrieved on top"


def test_dedup_skip():
    """Re-ingesting an already-completed book is skipped via the registry."""
    store = _store()
    reg = Registry()
    stats = ingest_book(_PDF_JUSTICE, store=store, embedder=_embedder, registry=reg, title="Justice")
    assert stats.status == "skipped", f"expected skipped, got {stats.status}"


def test_answer_short_circuit_when_no_results():
    """With no retrieved sources, answer is ungrounded and makes no API call."""
    ans = answer_question("anything", [], model=None)
    assert ans.grounded is False and ans.citations == []


def test_api_endpoints():
    """FastAPI routes against real Qdrant with fakes injected into the lifespan."""
    import ingest.embed as embed_mod
    import retrieval.rerank as rerank_mod

    embed_mod.BGEM3Embedder = FakeEmbedder      # lifespan builds these
    rerank_mod.Reranker = FakeReranker
    install_fake_llm("According to the books, justice is fairness under law [1].")

    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}

        st = client.get("/status").json()
        assert st["books"] >= 2 and st["chunks"] >= 1, st

        r = client.post("/query", json={"question": "justice fairness law"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert "answer" in body and "sources" in body and "citations" in body
        assert body["sources"], "query returned no sources"
        assert body["grounded"] is True

        # Directory of already-ingested PDFs -> all reported as skipped.
        r2 = client.post("/ingest", json={"path": str(_TMP), "recursive": False})
        assert r2.status_code == 200, r2.text
        assert r2.json()["skipped"], "expected already-ingested files to be skipped"

        # Web chatbot UI is served at /.
        ui = client.get("/")
        assert ui.status_code == 200 and "Maktabah" in ui.text, "chat UI not served"

        # /chat single turn -> grounded, cited, no condensation (search == question).
        c1 = client.post("/chat", json={"messages": [{"role": "user", "content": "justice fairness law"}]})
        assert c1.status_code == 200, c1.text
        cb = c1.json()
        assert cb["grounded"] is True and cb["citations"], cb
        assert cb["search_query"] == "justice fairness law", cb["search_query"]
        assert cb["sources"]

        # /chat multi-turn (condense disabled) -> still answers grounded.
        c2 = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": "Tell me about justice"},
                {"role": "assistant", "content": "It concerns fairness [1]."},
                {"role": "user", "content": "and the rule of law?"},
            ],
            "condense": False,
        })
        assert c2.status_code == 200, c2.text


def test_condense_query():
    """Follow-ups are rewritten into a standalone query; single turns are not."""
    from llm.chat import condense_query

    install_fake_llm("standalone rewritten query about law")
    # Single user turn -> returned verbatim, no LLM call.
    assert condense_query([{"role": "user", "content": "hello"}]) == "hello"
    # Multi-turn -> uses the (fake) LLM rewrite.
    msgs = [
        {"role": "user", "content": "about justice"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "and law?"},
    ]
    assert condense_query(msgs) == "standalone rewritten query about law"


def test_enqueue_book():
    """Real RQ enqueue (validates the args/kwargs fix); inspect the queued job."""
    from ingest.worker import enqueue_book, get_queue

    job_id = enqueue_book(str(_PDF_JUSTICE), title="X", author="Y")
    assert isinstance(job_id, str) and job_id
    job = get_queue().fetch_job(job_id)
    assert job is not None
    assert tuple(job.args) == (str(_PDF_JUSTICE),), job.args
    assert job.kwargs == {"title": "X", "author": "Y"}, job.kwargs
    assert job.func_name.endswith("ingest_book_job"), job.func_name


TESTS = [
    test_native_pdf_extraction,
    test_ingest_native_and_query,
    test_ingest_scanned_with_ocr,
    test_dedup_skip,
    test_answer_short_circuit_when_no_results,
    test_condense_query,
    test_api_endpoints,
    test_enqueue_book,
]


def main() -> int:
    print("Setting up (reset Qdrant collection + registry, generate PDFs)...")
    setup()
    failed = 0
    for fn in TESTS:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {fn.__name__} -> {type(exc).__name__}: {exc}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failed}/{len(TESTS)} integration tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
