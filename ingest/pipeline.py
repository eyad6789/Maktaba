"""Single-book ingestion orchestration.

`ingest_book` runs the full flow for one PDF:

    hash + dedup -> open -> build BookMeta -> classify each page ->
    extract native text / OCR scanned pages -> normalize -> chunk ->
    embed -> upsert into Qdrant -> mark completed in the registry.

OCR failures are captured per page (`IngestStats.failed_pages`) and never
abort the whole book; any unexpected error marks the book failed in the
registry and returns an `IngestStats(status="failed", ...)`. An optional
`on_progress(stage, current, total)` callback surfaces live progress (per-page
extraction, then the coarse summarize/embed/upsert steps) and is wrapped so a
failing callback can never fail ingestion.

Heavy libraries (FlagEmbedding, OCR backends, qdrant_client) are pulled in by
the collaborating objects passed in or imported lazily, so importing this
module on a CPU-only box stays cheap.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from config import settings
from core.logging import get_logger
from core.models import (
    BookMeta,
    Chunk,
    Embedding,
    IngestStats,
    PageContent,
    PageKind,
)

if TYPE_CHECKING:  # pragma: no cover - import only for type checkers
    import fitz

    from ingest.embed import BGEM3Embedder
    from ingest.ocr import OCRBackend
    from ingest.registry import Registry
    from retrieval.search import QdrantStore

logger = get_logger(__name__)

# Stable namespace for deriving a book_id from a file hash (uuid5).
NAMESPACE_URL = uuid.NAMESPACE_URL


def derive_book_id(file_hash: str) -> str:
    """Deterministic book id for a file hash: ``uuid5(NAMESPACE_URL, file_hash)``.

    Shared by the pipeline, the RQ worker (job metadata) and the upload
    endpoint so every component maps the same file to the same id without
    running the pipeline first.
    """
    return str(uuid.uuid5(NAMESPACE_URL, file_hash))


def build_book_meta(
    path: str | Path,
    file_hash: str,
    num_pages: int,
    title: str | None,
    author: str | None,
) -> BookMeta:
    """Construct a :class:`BookMeta` for one book.

    ``book_id`` is deterministic: ``uuid5(NAMESPACE_URL, file_hash)`` so the same
    file always maps to the same id across runs. ``title`` defaults to the file
    stem when not supplied. ``language`` is filled in later from page content.
    """
    src = Path(path)
    book_id = derive_book_id(file_hash)
    resolved_title = (title or "").strip() or src.stem
    return BookMeta(
        book_id=book_id,
        title=resolved_title,
        author=author,
        language=None,
        source_path=str(src),
        num_pages=num_pages,
        file_hash=file_hash,
    )


def _aggregate_language(pages: list[PageContent]) -> str | None:
    """Derive a book-level language from per-page detections.

    Returns ``"ar"``/``"en"`` when one language clearly dominates, ``"mixed"``
    when both are well represented, or ``None`` when nothing is detectable.
    """
    counts: dict[str, int] = {}
    for page in pages:
        lang = page.lang
        if not lang or lang == "unknown":
            continue
        counts[lang] = counts.get(lang, 0) + 1
    if not counts:
        return None
    if "mixed" in counts and counts["mixed"] >= max(counts.values()):
        return "mixed"
    ar = counts.get("ar", 0)
    en = counts.get("en", 0)
    total = ar + en
    if total == 0:
        # Some other single language detected; report the most common label.
        return max(counts, key=lambda k: counts[k])
    if ar and en and min(ar, en) >= 0.25 * total:
        return "mixed"
    return "ar" if ar >= en else "en"


def _extract_page(
    page: "fitz.Page",
    page_number: int,
    ocr: "OCRBackend | None",
    failed_pages: list[int],
) -> PageContent | None:
    """Classify and extract a single page into a :class:`PageContent`.

    Returns ``None`` for empty pages or pages whose extraction yielded no usable
    text. OCR exceptions are swallowed: the page number is recorded in
    ``failed_pages`` and ``None`` is returned so the book keeps ingesting.
    """
    from ingest import classify, extract, normalize

    kind = classify.classify_page(page, min_chars=settings.min_native_chars)

    if kind is PageKind.EMPTY:
        return None

    raw_text = ""
    if kind is PageKind.NATIVE:
        raw_text = extract.extract_native_text(page)
    elif kind is PageKind.SCANNED:
        if ocr is None:
            logger.warning(
                "page %d is scanned but no OCR backend available; skipping",
                page_number,
            )
            failed_pages.append(page_number)
            return None
        try:
            image = extract.render_page_image(page, dpi=settings.ocr_dpi)
            raw_text = ocr.ocr_image(image)
        except Exception as exc:  # noqa: BLE001 - per-page isolation by design
            logger.warning("OCR failed on page %d: %s", page_number, exc)
            failed_pages.append(page_number)
            return None

    text = normalize.normalize_text(raw_text or "")
    if not text.strip():
        # Native page that classified as having text but normalized to nothing,
        # or an OCR result that came back blank: nothing retrievable here.
        return None

    lang = normalize.detect_lang(text)
    return PageContent(page_number=page_number, text=text, kind=kind, lang=lang)


def _embed_chunks(
    embedder: "BGEM3Embedder", chunks: list[Chunk]
) -> list[Embedding]:
    """Embed chunk texts, preserving order and 1:1 correspondence to chunks."""
    texts = [chunk.text for chunk in chunks]
    embeddings = embedder.embed_documents(texts)
    if len(embeddings) != len(chunks):
        raise RuntimeError(
            f"embedder returned {len(embeddings)} embeddings for "
            f"{len(chunks)} chunks"
        )
    return embeddings


def _build_summary_nodes(
    doc: "fitz.Document",
    pages: list[PageContent],
    chunks: list[Chunk],
    book: BookMeta,
    num_pages: int,
) -> list[Chunk]:
    """Build the comprehension layer (chapter + book summaries) for one book.

    Detects structure from the PDF table of contents and summarizes with the
    small ``summary_model``. Best-effort: any failure (no model server, etc.)
    logs and returns ``[]`` so the book still ingests with its raw passages.
    """
    try:
        from ingest import structure, summarize

        toc = structure.extract_toc(doc)
        sections = structure.detect_structure(pages, toc, num_pages)
        nodes = summarize.build_summary_nodes(
            sections, chunks, book, model=settings.summary_model
        )
        logger.info(
            "Comprehension layer: %d summary node(s) for %s",
            len(nodes),
            book.book_id,
        )
        return nodes
    except Exception as exc:  # noqa: BLE001 - comprehension is best-effort
        logger.warning(
            "Comprehension layer failed for %s: %s; ingesting passages only",
            book.book_id,
            exc,
        )
        return []


def ingest_book(
    path: str | Path,
    *,
    store: "QdrantStore",
    embedder: "BGEM3Embedder",
    registry: "Registry",
    ocr: "OCRBackend | None" = None,
    title: str | None = None,
    author: str | None = None,
    force: bool = False,
    on_progress: Callable[[str, int, int], None] | None = None,
) -> IngestStats:
    """Ingest a single PDF book end to end.

    Args:
        path: Path to the PDF file.
        store: Qdrant store used to upsert chunk vectors.
        embedder: BGE-M3 embedder producing dense + sparse vectors.
        registry: SQLite registry for dedup and status tracking.
        ocr: OCR backend for scanned pages. Lazily created (factory) if any
            scanned page is encountered and none is supplied.
        title: Optional human title; defaults to the file stem.
        author: Optional author.
        force: Re-ingest even if this file was already ingested. Needed to
            rebuild the comprehension layer over an existing corpus. Upserts are
            idempotent (deterministic ids), so this overwrites rather than
            duplicates.
        on_progress: Optional ``(stage, current, total)`` callback for live
            progress reporting (stages: ``extracting``/``summarizing``/
            ``embedding``/``upserting``/``done``). Purely advisory — any
            exception it raises is swallowed and can never fail ingestion.

    Returns:
        An :class:`IngestStats` describing the outcome. ``status`` is
        ``"skipped"`` when the file was already ingested, ``"failed"`` on an
        unexpected error (with ``error`` set), otherwise ``"completed"``.
    """
    src = Path(path)
    file_hash = registry.compute_hash(src)

    def _progress(stage: str, current: int, total: int) -> None:
        """Invoke ``on_progress``; a failing callback must never abort the book."""
        if on_progress is None:
            return
        try:
            on_progress(stage, current, total)
        except Exception:  # noqa: BLE001 - progress is advisory by design
            logger.debug("progress callback failed at stage %r", stage, exc_info=True)

    if not force and registry.is_ingested(file_hash):
        book_id = derive_book_id(file_hash)
        logger.info("skipping already-ingested book: %s (%s)", src, book_id)
        return IngestStats(
            book_id=book_id,
            title=(title or "").strip() or src.stem,
            status="skipped",
        )

    book: BookMeta | None = None
    doc = None
    try:
        from ingest import extract

        doc = extract.open_pdf(src)
        num_pages = doc.page_count

        book = build_book_meta(src, file_hash, num_pages, title, author)
        registry.mark_started(book)
        logger.info(
            "ingesting %s -> book_id=%s (%d pages)", src, book.book_id, num_pages
        )

        failed_pages: list[int] = []
        pages: list[PageContent] = []
        ocr_backend = ocr

        for index in range(num_pages):
            page = doc.load_page(index)
            page_number = index + 1
            _progress("extracting", page_number, num_pages)

            # Lazily acquire an OCR backend only when a scanned page needs it.
            if ocr_backend is None:
                from ingest import classify

                if (
                    classify.classify_page(
                        page, min_chars=settings.min_native_chars
                    )
                    is PageKind.SCANNED
                ):
                    from ingest.ocr import get_ocr_backend

                    logger.info(
                        "scanned page detected; loading OCR backend '%s'",
                        settings.ocr_backend,
                    )
                    ocr_backend = get_ocr_backend(settings.ocr_backend)

            page_content = _extract_page(
                page, page_number, ocr_backend, failed_pages
            )
            if page_content is not None:
                pages.append(page_content)

        native_pages = sum(1 for p in pages if p.kind is PageKind.NATIVE)
        scanned_pages = sum(1 for p in pages if p.kind is PageKind.SCANNED)
        book.language = _aggregate_language(pages)

        from ingest import chunk as chunk_mod

        chunks = chunk_mod.chunk_pages(
            pages,
            book,
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
            min_tokens=settings.chunk_min_tokens,
        )

        # Build the hierarchical comprehension layer (chapter + book summaries)
        # when enabled. Summary nodes are Chunks too, so they ride the same
        # embed + upsert path and become retrievable alongside raw passages.
        summary_nodes: list[Chunk] = []
        if settings.enable_comprehension and chunks:
            _progress("summarizing", 0, 1)
            summary_nodes = _build_summary_nodes(doc, pages, chunks, book, num_pages)

        num_chunks = 0
        if chunks:
            nodes = chunks + summary_nodes
            _progress("embedding", 0, 1)
            embeddings = _embed_chunks(embedder, nodes)
            _progress("upserting", 0, 1)
            store.ensure_collection()
            store.upsert_chunks(nodes, embeddings)
            num_chunks = len(chunks)
        else:
            logger.warning(
                "book %s produced no chunks (pages=%d, failed=%d)",
                book.book_id,
                len(pages),
                len(failed_pages),
            )

        stats = IngestStats(
            book_id=book.book_id,
            title=book.title,
            num_pages=num_pages,
            native_pages=native_pages,
            scanned_pages=scanned_pages,
            failed_pages=failed_pages,
            num_chunks=num_chunks,
            num_summary_nodes=len(summary_nodes),
            status="completed",
        )
        registry.mark_completed(stats)
        logger.info(
            "completed %s: %d chunks, %d native, %d scanned, %d failed pages",
            book.book_id,
            num_chunks,
            native_pages,
            scanned_pages,
            len(failed_pages),
        )
        _progress("done", 1, 1)
        return stats

    except Exception as exc:  # noqa: BLE001 - top-level book guard
        logger.exception("ingestion failed for %s: %s", src, exc)
        if book is not None:
            try:
                registry.mark_failed(book.book_id, str(exc))
            except Exception:  # noqa: BLE001 - best-effort status update
                logger.exception("failed to mark book %s as failed", book.book_id)
        book_id = book.book_id if book is not None else derive_book_id(file_hash)
        return IngestStats(
            book_id=book_id,
            title=(book.title if book is not None else None)
            or (title or "").strip()
            or src.stem,
            status="failed",
            error=str(exc),
        )
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:  # noqa: BLE001 - best-effort cleanup
                pass
