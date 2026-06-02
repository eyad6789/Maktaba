"""Sentence/paragraph-aware chunking of extracted book pages.

Turns a book's per-page text (`list[PageContent]`) into retrievable `Chunk`s of
roughly `target_tokens` each, with `overlap_tokens` of trailing context carried
into the next chunk. Splitting respects paragraph and sentence boundaries and
never crosses the book (one call == one book).

Token heuristic
---------------
We do NOT load a tokenizer here (keeps this module light + CPU-importable).
Instead we approximate token count as ``max(1, len(s) // 4)`` — i.e. ~4
characters per token. This is a deliberate, language-agnostic estimate that
works reasonably for both Latin script and Arabic (where average word length
and BPE merge behaviour land near the same ratio for BGE-M3). It is only used
to decide chunk boundaries, so a rough estimate is sufficient.
"""

from __future__ import annotations

import re
import uuid

from config import settings
from core.logging import get_logger
from core.models import BookMeta, Chunk, PageContent

logger = get_logger(__name__)

# Sentence-final punctuation: Latin (. ! ?), Arabic question mark (؟),
# Arabic full stop / Urdu-style (۔), and the CJK/full-width stop (。) which
# sometimes appears in mixed scans. We keep the delimiter attached to the
# sentence it terminates by splitting *after* the punctuation run.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[\.!\?؟۔。])\s+")
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n+")
_WHITESPACE = re.compile(r"\s+")


def approx_tokens(text: str) -> int:
    """Approximate token count for *text* using the ~4-chars/token heuristic.

    Always returns at least 1 for non-empty input so a segment never counts as
    zero-cost when deciding chunk boundaries.
    """
    n = len(text)
    if n == 0:
        return 0
    return max(1, n // 4)


def _segment_pages(pages: list[PageContent]) -> list[tuple[str, int]]:
    """Flatten pages into ``(segment_text, page_number)`` units.

    Each page's text is split on blank-line paragraph breaks and then on
    sentence-final punctuation. Empty/whitespace-only segments are dropped.
    Page order (and thus reading order) is preserved.
    """
    segments: list[tuple[str, int]] = []
    for page in pages:
        raw = page.text or ""
        if not raw.strip():
            continue
        for paragraph in _PARAGRAPH_BOUNDARY.split(raw):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            for sentence in _SENTENCE_BOUNDARY.split(paragraph):
                # Collapse internal whitespace/newlines within a sentence so the
                # stored chunk text reads cleanly; final normalization happens
                # later via normalize_text.
                cleaned = _WHITESPACE.sub(" ", sentence).strip()
                if cleaned:
                    segments.append((cleaned, page.page_number))
    return segments


def _overlap_tail(
    segments: list[tuple[str, int]], overlap_tokens: int
) -> list[tuple[str, int]]:
    """Return the trailing segments whose combined tokens cover *overlap_tokens*.

    Walks backwards accumulating segments until the overlap budget is met, then
    returns them in original (forward) order. Returns an empty list when
    *overlap_tokens* <= 0.
    """
    if overlap_tokens <= 0 or not segments:
        return []
    tail: list[tuple[str, int]] = []
    acc = 0
    for seg in reversed(segments):
        tail.append(seg)
        acc += approx_tokens(seg[0])
        if acc >= overlap_tokens:
            break
    tail.reverse()
    return tail


def _build_chunk(
    book: BookMeta,
    idx: int,
    segments: list[tuple[str, int]],
) -> Chunk | None:
    """Assemble one `Chunk` from member *segments*, or None if empty.

    Joins segment texts with single spaces, normalizes the result, derives the
    page span from member pages, and computes a stable uuid5 chunk id.
    """
    if not segments:
        return None

    # Lazy import: normalize lives in a sibling module and pulls in optional
    # language-detection deps; importing here keeps this module light to import.
    from ingest.normalize import detect_lang, normalize_text

    joined = " ".join(text for text, _ in segments).strip()
    if not joined:
        return None

    normalized = normalize_text(joined)
    if not normalized.strip():
        return None

    pages = [page for _, page in segments]
    page_start = min(pages)
    page_end = max(pages)

    try:
        lang = detect_lang(normalized)
    except Exception:  # pragma: no cover - language detection is best-effort
        lang = None

    chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{book.book_id}:{idx}"))

    return Chunk(
        chunk_id=chunk_id,
        book_id=book.book_id,
        title=book.title,
        author=book.author,
        text=normalized,
        page_start=page_start,
        page_end=page_end,
        chunk_index=idx,
        lang=lang,
        token_count=approx_tokens(normalized),
    )


def chunk_pages(
    pages: list[PageContent],
    book: BookMeta,
    *,
    target_tokens: int = settings.chunk_target_tokens,
    overlap_tokens: int = settings.chunk_overlap_tokens,
    min_tokens: int = settings.chunk_min_tokens,
) -> list[Chunk]:
    """Split a book's page texts into overlapping, sentence-aware `Chunk`s.

    Segments are accumulated until the running token estimate reaches
    *target_tokens*; the chunk is emitted and the next chunk seeds with roughly
    *overlap_tokens* of trailing segments for context continuity. Chunking never
    crosses the book (this call handles exactly one book).

    The final chunk is dropped when its estimated tokens fall below
    *min_tokens*, unless it would be the only chunk for the book.

    Args:
        pages: Per-page extracted/OCR'd text in 1-based reading order.
        book: Identity/metadata for the book these pages belong to.
        target_tokens: Approximate target size of each chunk.
        overlap_tokens: Approximate trailing context carried into the next chunk.
        min_tokens: Minimum size for a trailing chunk to be kept.

    Returns:
        Ordered list of `Chunk`s with sequential ``chunk_index`` starting at 0.
    """
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")
    # Clamp overlap so it cannot starve forward progress.
    if overlap_tokens >= target_tokens:
        overlap_tokens = max(0, target_tokens // 2)

    segments = _segment_pages(pages)
    if not segments:
        logger.info("No text segments to chunk for book_id=%s", book.book_id)
        return []

    chunks: list[Chunk] = []
    idx = 0
    current: list[tuple[str, int]] = []
    current_tokens = 0

    for segment in segments:
        seg_tokens = approx_tokens(segment[0])
        current.append(segment)
        current_tokens += seg_tokens

        if current_tokens >= target_tokens:
            chunk = _build_chunk(book, idx, current)
            if chunk is not None:
                chunks.append(chunk)
                idx += 1
            # Seed the next chunk with the overlapping tail of this one.
            tail = _overlap_tail(current, overlap_tokens)
            current = list(tail)
            current_tokens = sum(approx_tokens(t) for t, _ in current)

    # Flush any remainder. Drop it if it is too small AND we already have chunks
    # (i.e. it is purely the overlap tail or a sub-min straggler).
    if current:
        remainder_tokens = sum(approx_tokens(t) for t, _ in current)
        if chunks and remainder_tokens < min_tokens:
            logger.debug(
                "Dropping trailing %d-token remainder for book_id=%s",
                remainder_tokens,
                book.book_id,
            )
        else:
            chunk = _build_chunk(book, idx, current)
            if chunk is not None:
                chunks.append(chunk)
                idx += 1

    logger.info(
        "Chunked book_id=%s: %d pages -> %d segments -> %d chunks "
        "(target=%d, overlap=%d, min=%d)",
        book.book_id,
        len(pages),
        len(segments),
        len(chunks),
        target_tokens,
        overlap_tokens,
        min_tokens,
    )
    return chunks
