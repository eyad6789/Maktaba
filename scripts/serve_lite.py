"""Run the full app locally on a CPU box — DEV / DEMO launcher.

Brings the system up without a GPU or the BGE-M3 / reranker downloads:
  * deterministic CPU stand-ins for the embedder + reranker (scripts.lite_backends),
  * a seeded bilingual (Arabic + English) sample library so there's content,
  * real Qdrant for storage + hybrid search,
  * real Claude answers when ANTHROPIC_API_KEY is set, otherwise a graceful
    retrieval-only mode that returns the matching passages.

Requires Qdrant reachable at QDRANT_URL (e.g. `docker compose up -d qdrant`).

    python -m scripts.serve_lite            # http://localhost:8000

This is for local demos/development only — retrieval quality is approximate.
For production use the real backends on the GPU server (see README).
"""

from __future__ import annotations

import hashlib
import os
import uuid

# Local lite-mode defaults (set before importing config).
import importlib.util as _ilu
from pathlib import Path as _Path

os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("RERANKER_DEVICE", "cpu")
os.environ.setdefault("REGISTRY_DB", "data/registry/lite.db")
os.environ.setdefault("LOG_LEVEL", "INFO")

# Embedder: prefer the REAL multilingual semantic model (sentence-transformers
# MiniLM, 384-d) for good retrieval; fall back to the lexical hashing stand-in.
USE_ST = (
    os.environ.get("LITE_EMBED", "st").lower() != "hash"
    and _ilu.find_spec("sentence_transformers") is not None
    and any((_Path.home() / ".cache/huggingface/hub").glob(
        "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2"))
)
if USE_ST:
    os.environ.setdefault("EMBEDDING_DIM", "384")     # MiniLM dimension
    os.environ.setdefault("QDRANT_COLLECTION", "books_demo")
else:
    os.environ.setdefault("QDRANT_COLLECTION", "books_lite")

# How to answer locally:
#   LITE_LLM=off          (default) retrieval-only — show the matching passages
#   LITE_LLM=transformers run a small local model in-process (CPU; slow but real)
#   LITE_LLM=ollama       use a local Ollama server (offline, fast)
LITE_LLM = os.environ.get("LITE_LLM", "off").lower()
if LITE_LLM == "transformers":
    os.environ.setdefault("LLM_BACKEND", "transformers")
    os.environ.setdefault("LOCAL_LLM_DEVICE", "cpu")
    os.environ.setdefault("LOCAL_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
elif LITE_LLM == "ollama":
    os.environ.setdefault("LLM_BACKEND", "openai")
    os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:11434/v1")
    os.environ.setdefault("OPENAI_MODEL", "qwen2.5")

from config import settings  # noqa: E402
from core.logging import get_logger  # noqa: E402
from core.models import (  # noqa: E402
    Answer,
    BookMeta,
    Citation,
    IngestStats,
    PageContent,
    PageKind,
)

logger = get_logger("serve_lite")

HOST = os.environ.get("API_HOST", "0.0.0.0")
PORT = int(os.environ.get("API_PORT", "8000"))

# --- sample bilingual library ----------------------------------------------

SAMPLE_BOOKS: list[tuple[str, str, str, list[str]]] = [
    (
        "On Justice and Law",
        "A. Reader",
        "en",
        [
            "On Justice and Law is organized into three chapters. Chapter 1, 'Fairness', "
            "defines justice as the fair and impartial treatment of every person. Chapter "
            "2, 'The Rule of Law', argues that no one is above the law. Chapter 3, 'The "
            "Courts', describes how impartial judges apply the law to reach verdicts. The "
            "book's central argument is that a legal order earns the trust of its people "
            "only when fairness, equality, and accountability are upheld together.",
            "Chapter 1 — Fairness. Justice begins with fairness: treating like cases alike "
            "and giving each person their due. A just society does not privilege the "
            "powerful; it measures every citizen by the same standard. The main idea of "
            "this chapter is that fairness and equality before the law are the foundation "
            "of justice.",
            "Chapter 2 — The Rule of Law. The rule of law means that laws, not individuals, "
            "govern, and that even rulers are bound by them. Chapter 3 — The Courts. "
            "Independent judges weigh evidence impartially so that verdicts rest on reason "
            "and law rather than power or wealth, making accountability real.",
        ],
    ),
    (
        "A Primer on Astronomy",
        "A. Reader",
        "en",
        [
            "A Primer on Astronomy is organized into three chapters. Chapter 1, 'The Solar "
            "System', introduces the Sun, the eight planets, and their moons. Chapter 2, "
            "'Stars', explains how stars are born, shine through nuclear fusion, and "
            "eventually die. Chapter 3, 'Galaxies and the Cosmos', surveys the Milky Way, "
            "other galaxies, and the expanding universe. The book's central theme is that "
            "the same physical laws govern planets, stars, and entire galaxies.",
            "Chapter 1 — The Solar System. The Sun is an average star at the centre of our "
            "system, holding the planets in orbit by gravity. The four inner planets are "
            "small and rocky; the four outer planets are gas and ice giants; moons, "
            "asteroids, and comets complete the family. The main idea of this chapter is "
            "that the Solar System is a gravitationally bound system of diverse bodies "
            "orbiting one ordinary star.",
            "Chapter 2 — Stars. Stars form when clouds of gas collapse until fusion "
            "ignites, converting hydrogen into helium and releasing light; massive stars "
            "end as supernovae. Chapter 3 — Galaxies and the Cosmos. The Milky Way holds "
            "hundreds of billions of stars, beyond it lie countless galaxies, and "
            "observations show the universe is expanding.",
        ],
    ),
    (
        "في العدالة والقانون",
        "كاتب",
        "ar",
        [
            "ينقسم كتاب «في العدالة والقانون» إلى ثلاثة فصول. الفصل الأول، «الإنصاف»، يعرّف "
            "العدالة بأنها معاملة كل إنسان معاملةً منصفةً ومتساوية. الفصل الثاني، «سيادة "
            "القانون»، يؤكد أن لا أحد فوق القانون. الفصل الثالث، «المحاكم»، يصف كيف يطبّق "
            "القضاة المستقلون القانون للوصول إلى الأحكام. وحجة الكتاب المحورية أن النظام "
            "القانوني لا ينال ثقة الناس إلا حين يجتمع الإنصاف والمساواة والمحاسبة.",
            "الفصل الأول — الإنصاف. تبدأ العدالة بالإنصاف: معاملة الحالات المتماثلة بالمثل "
            "وإعطاء كل ذي حقٍّ حقّه. المجتمع العادل لا يحابي الأقوياء بل يقيس كل مواطن "
            "بالمعيار نفسه. الفكرة الرئيسية لهذا الفصل أن الإنصاف والمساواة أمام القانون "
            "أساس العدالة.",
            "الفصل الثاني — سيادة القانون. تعني سيادة القانون أن تحكم القوانين لا الأفراد، "
            "وأن يكون الحكّام أنفسهم خاضعين لها. الفصل الثالث — المحاكم. يزن القضاة "
            "المستقلون الأدلة بحياد كي تقوم الأحكام على العقل والقانون لا على القوة أو "
            "المال، فتتحقق المحاسبة.",
        ],
    ),
    (
        "مدخل إلى علم الفلك",
        "كاتب",
        "ar",
        [
            "ينقسم كتاب «مدخل إلى علم الفلك» إلى ثلاثة فصول. الفصل الأول، «المجموعة "
            "الشمسية»، يعرّف بالشمس والكواكب الثمانية وأقمارها. الفصل الثاني، «النجوم»، "
            "يشرح كيف تولد النجوم وتشعّ عبر الاندماج النووي ثم تموت. الفصل الثالث، «المجرات "
            "والكون»، يستعرض مجرّة درب التبّانة وبقيّة المجرّات وتمدّد الكون. الفكرة "
            "المحورية للكتاب أن القوانين الفيزيائية نفسها تحكم الكواكب والنجوم والمجرّات.",
            "الفصل الأول — المجموعة الشمسية. الشمس نجمٌ متوسّط في مركز مجموعتنا، تُمسك "
            "بالكواكب في مداراتها بفعل الجاذبية. الكواكب الأربعة الداخلية صخرية صغيرة، "
            "والأربعة الخارجية عمالقة غازية وجليدية. الفكرة الرئيسية لهذا الفصل أن المجموعة "
            "الشمسية منظومة مترابطة بالجاذبية من أجرامٍ متنوّعة تدور حول نجمٍ واحدٍ عادي.",
            "الفصل الثاني — النجوم. تتكوّن النجوم حين تنهار سُحب الغاز حتى يشتعل الاندماج "
            "فيحوّل الهيدروجين إلى هيليوم باعثًا الضوء. الفصل الثالث — المجرّات. تضمّ مجرّة "
            "درب التبّانة مئات المليارات من النجوم، وخلفها مجرّاتٌ لا تُحصى، وتدلّ الأرصاد "
            "على أن الكون يتمدّد.",
        ],
    ),
]


def seed_library(store, embedder, registry) -> int:
    """Upsert the sample books into Qdrant if the collection is empty."""
    try:
        if store.count() > 0:
            logger.info("Library already populated (%d passages); skipping seed", store.count())
            return 0
    except Exception:
        pass

    from ingest.chunk import chunk_pages

    total = 0
    for title, author, lang, pages in SAMPLE_BOOKS:
        file_hash = hashlib.sha256(title.encode("utf-8")).hexdigest()
        if registry.is_ingested(file_hash):
            continue
        book_id = str(uuid.uuid5(uuid.NAMESPACE_URL, file_hash))
        book = BookMeta(
            book_id=book_id,
            title=title,
            author=author,
            language=lang,
            source_path=f"seed://{title}",
            num_pages=len(pages),
            file_hash=file_hash,
        )
        registry.mark_started(book)
        page_contents = [
            PageContent(page_number=i + 1, text=t, kind=PageKind.NATIVE, lang=lang)
            for i, t in enumerate(pages)
        ]
        chunks = chunk_pages(page_contents, book)
        embeddings = embedder.embed_documents([c.text for c in chunks])
        store.ensure_collection()
        n = store.upsert_chunks(chunks, embeddings)
        registry.mark_completed(
            IngestStats(
                book_id=book_id, title=title, num_pages=len(pages),
                native_pages=len(pages), num_chunks=n, status="completed",
            )
        )
        total += n
        logger.info("Seeded %r (%s) -> %d chunk(s)", title, lang, n)
    return total


# --- retrieval-only fallback (no API key) -----------------------------------


def _last_user(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").strip()
    return ""


def _retrieval_only(results) -> Answer:
    if not results:
        return Answer(
            answer="No matching passages were found in the library.",
            citations=[], sources=[], model="lite-no-llm", grounded=False,
        )
    top = results[:3]
    lines = [
        "(No ANTHROPIC_API_KEY set — showing the most relevant passages from your "
        "books. Add your key and restart for a written AI answer.)",
        "",
    ]
    citations: list[Citation] = []
    for i, s in enumerate(top, 1):
        pg = f"p.{s.page_start}" if s.page_start == s.page_end else f"pp.{s.page_start}-{s.page_end}"
        snippet = s.text if len(s.text) <= 260 else s.text[:260] + "…"
        lines.append(f"[{i}] {s.title} ({pg}): {snippet}")
        citations.append(
            Citation(title=s.title, author=s.author, page_start=s.page_start,
                     page_end=s.page_end, book_id=s.book_id)
        )
    return Answer(answer="\n".join(lines), citations=citations, sources=results,
                  model="lite-no-llm", grounded=False)


def main() -> None:
    # 1. Swap in CPU backends (the API lifespan builds these). Prefer the REAL
    #    semantic embedder + reranker; fall back to the lexical hashing stand-in.
    import ingest.embed as embed_mod
    import retrieval.rerank as rerank_mod

    if USE_ST:
        from scripts.lite_backends import SemanticReranker, SentenceTransformerEmbedder

        emb_class = SentenceTransformerEmbedder

        class _STReranker(SemanticReranker):  # lifespan calls Reranker() with no args
            def __init__(self, *_, **__):
                super().__init__(SentenceTransformerEmbedder())

        rer_class = _STReranker
        retrieval_label = "semantic (multilingual MiniLM, 384-d) + cosine rerank"
    else:
        from scripts.lite_backends import HashingEmbedder, OverlapReranker

        emb_class = HashingEmbedder
        rer_class = OverlapReranker
        retrieval_label = "lexical hashing (install sentence-transformers for semantic)"

    embed_mod.BGEM3Embedder = emb_class
    rerank_mod.Reranker = rer_class

    # 2. Seed the sample library into Qdrant.
    from ingest.registry import Registry
    from retrieval.search import QdrantStore

    store = QdrantStore()
    store.ensure_collection()
    seeded = seed_library(store, emb_class(), Registry())
    logger.info("Library ready: %d total passages (seeded %d) | retrieval: %s",
                store.count(), seeded, retrieval_label)

    # 3. Answer mode. Default (LITE_LLM=off) degrades /chat and /query to
    #    retrieval-only; otherwise a local offline model answers for real.
    if LITE_LLM == "off":
        import llm.answer as answer_mod
        import llm.chat as chat_mod

        chat_mod.condense_query = lambda messages, **kw: _last_user(messages)
        chat_mod.chat_answer = lambda messages, results, **kw: _retrieval_only(results)
        answer_mod.answer_question = lambda question, results, **kw: _retrieval_only(results)
        logger.warning(
            "RETRIEVAL-ONLY mode (passages shown, no AI synthesis). "
            "Run with LITE_LLM=transformers or LITE_LLM=ollama for real offline answers."
        )
    else:
        logger.info("Offline LLM enabled: backend=%s model=%s", settings.llm_backend, settings.local_llm_model if settings.llm_backend == "transformers" else settings.openai_model)

    # 4. Serve.
    import uvicorn
    from api.main import app

    print("\n" + "=" * 60)
    print("  Maktabah is running (LITE / demo mode)")
    print(f"  → open http://localhost:{PORT} in your browser")
    print(f"  → {store.count()} passages across {len(SAMPLE_BOOKS)} sample books (AR + EN)")
    print(f"  → retrieval: {retrieval_label}")
    mode = "retrieval-only (set LITE_LLM=transformers|ollama for AI answers)" if LITE_LLM == "off" else f"offline LLM · {settings.llm_backend}"
    print(f"  → answer mode: {mode}")
    print("=" * 60 + "\n")
    uvicorn.run(app, host=HOST, port=PORT, log_level=settings.log_level.lower())


if __name__ == "__main__":
    main()
