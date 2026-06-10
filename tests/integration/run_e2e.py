"""End-to-end integration run against REAL Qdrant + Redis.

Exercises the actual pipeline / retrieval / answer / API / queue code with
deterministic test doubles for the GPU/cloud pieces (embedder, reranker, OCR,
Claude). Requires:

  * Qdrant reachable at QDRANT_URL (default http://localhost:6333)
  * Redis reachable at REDIS_URL  (set below to redis://localhost:6380/0)

Run:  .venv/bin/python -m tests.integration.run_e2e
"""

from __future__ import annotations

import json
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
os.environ.setdefault("CONVERSATIONS_DB", str(_TMP / "conversations.db"))
os.environ.setdefault("UPLOADS_DIR", str(_TMP / "uploads"))
os.environ.setdefault("LOG_LEVEL", "WARNING")
# Pin comprehension OFF regardless of the developer's .env (env vars beat the
# dotenv file): with it on, every ingest builds summary nodes via the REAL LLM
# engine before the fakes are installed, breaking determinism (and the
# comprehension test, which expects to be the only book with summary nodes).
os.environ.setdefault("ENABLE_COMPREHENSION", "false")

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
    install_fake_llm_stream,
    install_fake_llm_stream_rate_limited,
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
CHEMISTRY = (
    "Chemistry studies atoms, molecules, and chemical reactions. Acids and "
    "bases neutralize each other in solution. The periodic table organizes "
    "the chemical elements by atomic number, and chemical bonds hold the "
    "molecules of every compound together during reactions."
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
    """Reset Qdrant test collection + SQLite DBs + uploads, regenerate PDFs."""
    from qdrant_client import QdrantClient

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key)
    if client.collection_exists(settings.qdrant_collection):
        client.delete_collection(settings.qdrant_collection)
    # Both SQLite stores are isolated to the temp dir (REGISTRY_DB /
    # CONVERSATIONS_DB env vars above); wipe them plus their WAL sidecars.
    for db in (Path(settings.registry_db), Path(settings.conversations_db)):
        for suffix in ("", "-wal", "-shm"):
            sidecar = Path(str(db) + suffix)
            if sidecar.exists():
                sidecar.unlink()
    # Uploads land in the temp dir too (UPLOADS_DIR env var above).
    uploads = Path(settings.uploads_dir)
    if uploads.is_dir():
        for leftover in uploads.iterdir():
            if leftover.is_file():
                leftover.unlink()

    make_native_pdf(_PDF_JUSTICE, [JUSTICE])
    make_native_pdf(_PDF_ASTRO, [ASTRONOMY])
    make_scanned_pdf(_PDF_SCANNED)


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    """Parse an SSE body into ordered ``(event, data)`` pairs.

    Frames are separated by a blank line; each carries one ``event:`` line and
    one single-line JSON ``data:`` payload (see ``core/sse.py``).
    """
    events: list[tuple[str, dict]] = []
    for frame in body.split("\n\n"):
        if not frame.strip():
            continue
        event, data = None, None
        for line in frame.splitlines():
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data = json.loads(line[len("data: "):])
        assert event is not None and data is not None, f"malformed SSE frame: {frame!r}"
        events.append((event, data))
    return events


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


def test_force_reingest():
    """force=True re-ingests an already-completed book (completed, not skipped)."""
    store = _store()
    reg = Registry()
    skipped = ingest_book(_PDF_JUSTICE, store=store, embedder=_embedder, registry=reg, title="Justice")
    assert skipped.status == "skipped", skipped
    forced = ingest_book(
        _PDF_JUSTICE, store=store, embedder=_embedder, registry=reg, title="Justice", force=True
    )
    assert forced.status == "completed", forced
    assert forced.num_chunks >= 1


def test_comprehension_layer_and_global_route():
    """Ingest with the comprehension layer ON; summary nodes are built, are
    level-filterable in Qdrant, and the GLOBAL route surfaces them."""
    from retrieval.pipeline import retrieve_for_route
    from retrieval.route import Route

    pdf = _TMP / "comprehension.pdf"
    make_native_pdf(pdf, [JUSTICE, JUSTICE, JUSTICE, JUSTICE])

    store = _store()
    reg = Registry()
    install_fake_llm("This book examines justice, fairness, and the rule of law in society.")

    orig = settings.enable_comprehension
    settings.enable_comprehension = True
    try:
        stats = ingest_book(pdf, store=store, embedder=_embedder, registry=reg, title="On Justice")
    finally:
        settings.enable_comprehension = orig

    assert stats.status == "completed", stats
    assert stats.num_summary_nodes >= 1, f"no summary nodes built: {stats}"

    # The book overview node is retrievable and filterable by level.
    q = _embedder.embed_query("justice fairness law")
    summaries = store.hybrid_search(q, top_k=settings.search_top_k, levels=["book_summary"])
    assert summaries, "book_summary node not retrievable via levels filter"
    assert all(s.level == "book_summary" for s in summaries)
    assert summaries[0].book_id == stats.book_id

    # A passages-only search must NOT return the summary node.
    passages = store.hybrid_search(q, top_k=settings.search_top_k, levels=["passage"])
    assert passages and all(p.level == "passage" for p in passages)

    # GLOBAL route surfaces a summary node as part of the answer context.
    ctx = retrieve_for_route(
        "what is the main idea of this book", q, store, _reranker, Route.GLOBAL,
        book_ids=[stats.book_id],
    )
    assert ctx, "GLOBAL route returned no context"
    assert any(r.level in ("book_summary", "chapter_summary") for r in ctx), (
        "GLOBAL route did not surface a summary node"
    )

    # End to end through the API: a thematic question auto-routes GLOBAL and the
    # answer's sources include a summary node.
    import ingest.embed as embed_mod
    import retrieval.rerank as rerank_mod

    embed_mod.BGEM3Embedder = FakeEmbedder
    rerank_mod.Reranker = FakeReranker

    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        r = client.post("/query", json={
            "question": "what is the main idea of this book",
            "book_ids": [stats.book_id],
        })
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["grounded"] is True, body
        levels = {s.get("level") for s in body["sources"]}
        assert levels & {"book_summary", "chapter_summary"}, body["sources"]


def test_models_endpoint():
    """GET /models lists Auto (with its chain) plus every provider's row."""
    import ingest.embed as embed_mod
    import retrieval.rerank as rerank_mod

    embed_mod.BGEM3Embedder = FakeEmbedder
    rerank_mod.Reranker = FakeReranker

    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        r = client.get("/models")
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["default"] == "auto", body
        providers = body["providers"]
        assert providers and providers[0]["id"] == "auto", providers
        assert isinstance(providers[0]["chain"], list) and providers[0]["chain"], providers[0]
        for p in providers:
            for key in ("id", "label", "model", "available"):
                assert key in p, f"provider row missing {key!r}: {p}"


def test_upload_ingest_books_jobs_delete():
    """Dashboard round-trip: upload -> (sync) ingest -> list -> dedup -> delete."""
    import ingest.embed as embed_mod
    import retrieval.rerank as rerank_mod

    embed_mod.BGEM3Embedder = FakeEmbedder
    rerank_mod.Reranker = FakeReranker
    install_fake_llm("Chemistry studies atoms and molecules [1].")

    from fastapi.testclient import TestClient
    from api.main import app
    from ingest.worker import ingest_book_job

    pdf = _TMP / "chemistry.pdf"
    make_native_pdf(pdf, [CHEMISTRY])
    pdf_bytes = pdf.read_bytes()

    with TestClient(app) as client:
        baseline_chunks = _store().count()

        # Multipart upload -> streamed to uploads_dir, hashed, enqueued on RQ.
        r = client.post(
            "/upload",
            files={"file": ("chemistry.pdf", pdf_bytes, "application/pdf")},
            data={"title": "Chemistry"},
        )
        assert r.status_code == 200, r.text
        up = r.json()
        assert up["status"] == "queued", up
        assert up["job_id"] and up["book_id"], up
        assert up["size_bytes"] == len(pdf_bytes), up
        saved = Path(settings.uploads_dir) / up["filename"]
        assert saved.is_file(), f"upload not saved at {saved}"

        # Simulate the worker: run the job function synchronously (no RQ worker
        # in this suite); the fakes installed above stand in for GPU pieces.
        stats = ingest_book_job(str(saved), title="Chemistry")
        assert stats["status"] == "completed" and stats["num_chunks"] >= 1, stats
        assert stats["book_id"] == up["book_id"], (stats["book_id"], up["book_id"])

        # The book shows up completed with consistent corpus totals.
        books = client.get("/books")
        assert books.status_code == 200, books.text
        bb = books.json()
        row = next((b for b in bb["books"] if b["book_id"] == up["book_id"]), None)
        assert row is not None, bb
        assert row["status"] == "completed" and row["num_chunks"] >= 1, row
        assert bb["total_books"] == len(bb["books"]), bb
        assert bb["total_chunks"] == _store().count() > baseline_chunks, bb

        # Job polling: the real worker never ran, so only assert the shape.
        j = client.get(f"/jobs/{up['job_id']}")
        assert j.status_code == 200, j.text
        jb = j.json()
        assert jb["job_id"] == up["job_id"], jb
        assert jb["state"] in {"queued", "started", "finished", "failed", "not_found"}, jb

        # Re-uploading identical bytes is caught by content hash.
        r2 = client.post(
            "/upload",
            files={"file": ("chemistry.pdf", pdf_bytes, "application/pdf")},
            data={"title": "Chemistry"},
        )
        assert r2.status_code == 200, r2.text
        dup = r2.json()
        assert dup["status"] == "duplicate" and dup["book_id"] == up["book_id"], dup

        # Delete removes vectors, the registry row, and the uploaded file.
        d = client.delete(f"/books/{up['book_id']}")
        assert d.status_code == 200, d.text
        assert d.json()["deleted"] is True, d.text
        after = client.get("/books").json()
        assert up["book_id"] not in {b["book_id"] for b in after["books"]}, after
        assert _store().count() == baseline_chunks, "Qdrant count did not drop back"
        assert not saved.exists(), "uploaded source file not removed on delete"


def test_conversations_crud():
    """POST -> list -> PATCH rename -> GET detail -> DELETE -> 404."""
    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        r = client.post("/conversations", json={"title": "My research", "model": "auto"})
        assert r.status_code == 201, r.text
        conv = r.json()
        conv_id = conv["id"]
        assert conv_id and conv["title"] == "My research", conv
        assert conv["message_count"] == 0, conv

        listed = client.get("/conversations").json()["conversations"]
        assert conv_id in {c["id"] for c in listed}, listed

        p = client.patch(f"/conversations/{conv_id}", json={"title": "Renamed"})
        assert p.status_code == 200, p.text
        assert p.json()["title"] == "Renamed", p.text

        g = client.get(f"/conversations/{conv_id}")
        assert g.status_code == 200, g.text
        detail = g.json()
        assert detail["title"] == "Renamed" and detail["messages"] == [], detail

        d = client.delete(f"/conversations/{conv_id}")
        assert d.status_code == 200 and d.json() == {"deleted": True}, d.text
        assert client.get(f"/conversations/{conv_id}").status_code == 404


def test_chat_stream_happy_path():
    """POST /chat/stream: meta -> provider -> delta+ -> done, turns persisted."""
    import ingest.embed as embed_mod
    import retrieval.rerank as rerank_mod

    embed_mod.BGEM3Embedder = FakeEmbedder
    rerank_mod.Reranker = FakeReranker
    answer = "Justice is fairness under the law [1]."
    install_fake_llm(answer)  # condense_query still goes through complete()
    install_fake_llm_stream(answer, provider_id="gemini", model="fake-model", n_chunks=3)

    from fastapi.testclient import TestClient
    from api.main import app

    question = "justice fairness law"
    with TestClient(app) as client:
        with client.stream(
            "POST", "/chat/stream", json={"conversation_id": None, "message": question}
        ) as resp:
            assert resp.status_code == 200, resp.status_code
            assert resp.headers["content-type"].startswith("text/event-stream"), resp.headers
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        names = [name for name, _ in events]
        assert names[0] == "meta", names
        assert names[1] == "provider", names
        assert names[-1] == "done", names
        deltas = names[2:-1]
        assert deltas and set(deltas) == {"delta"}, names

        meta = events[0][1]
        conv_id = meta["conversation_id"]
        assert conv_id, meta
        assert isinstance(meta["sources"], list) and meta["sources"], meta

        assert events[1][1]["provider"] == "gemini", events[1][1]

        streamed = "".join(d["text"] for name, d in events if name == "delta")
        assert streamed == answer, streamed

        done = events[-1][1]
        assert done["conversation_id"] == conv_id, done
        assert done["provider"] == "gemini", done
        assert isinstance(done["citations"], list) and done["citations"], done
        assert done["grounded"] is True, done

        # Both turns persisted server-side; title derived from the user message.
        detail = client.get(f"/conversations/{conv_id}").json()
        msgs = detail["messages"]
        assert len(msgs) == 2, msgs
        assert msgs[0]["role"] == "user" and msgs[0]["content"] == question, msgs
        assert msgs[1]["role"] == "assistant" and msgs[1]["model"] == "gemini", msgs
        assert detail["title"] == question, detail


def test_chat_stream_pinned_rate_limit():
    """A pinned provider 429ing yields meta -> error (no done, no assistant turn)."""
    import ingest.embed as embed_mod
    import retrieval.rerank as rerank_mod

    embed_mod.BGEM3Embedder = FakeEmbedder
    rerank_mod.Reranker = FakeReranker
    install_fake_llm_stream_rate_limited("gemini")

    from fastapi.testclient import TestClient
    from api.main import app

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/chat/stream",
            json={"message": "justice fairness law", "provider": "gemini"},
        ) as resp:
            assert resp.status_code == 200, resp.status_code
            body = resp.read().decode("utf-8")

        events = _parse_sse(body)
        names = [name for name, _ in events]
        assert names == ["meta", "error"], names

        err = events[1][1]
        assert err["provider"] == "gemini", err
        assert err["reason"] == "rate_limit", err
        assert err["partial"] is False, err
        assert "rate limit" in err["message"].lower(), err

        # The user turn is persisted (so retry works); the assistant turn is not.
        conv_id = events[0][1]["conversation_id"]
        detail = client.get(f"/conversations/{conv_id}").json()
        msgs = detail["messages"]
        assert len(msgs) == 1 and msgs[0]["role"] == "user", msgs


TESTS = [
    test_native_pdf_extraction,
    test_ingest_native_and_query,
    test_ingest_scanned_with_ocr,
    test_dedup_skip,
    test_answer_short_circuit_when_no_results,
    test_condense_query,
    test_api_endpoints,
    test_enqueue_book,
    test_force_reingest,
    test_comprehension_layer_and_global_route,
    test_models_endpoint,
    test_upload_ingest_books_jobs_delete,
    test_conversations_crud,
    test_chat_stream_happy_path,
    test_chat_stream_pinned_rate_limit,
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
