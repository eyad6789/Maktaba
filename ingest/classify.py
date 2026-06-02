"""Page classification: decide whether a PDF page is native text, scanned, or empty.

The ingestion pipeline uses this to route each page either to fast native text
extraction (PyMuPDF) or to the (slow, GPU-bound) OCR backend. Pages with neither
meaningful text nor images are skipped as EMPTY.

`fitz` (PyMuPDF) is lightweight and may be imported at module top; it is the only
non-stdlib dependency here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF — lightweight, safe to import at module top

from config import settings
from core.logging import get_logger
from core.models import PageKind

if TYPE_CHECKING:  # pragma: no cover - typing only
    import fitz as _fitz  # noqa: F401

logger = get_logger(__name__)


def classify_page(page: "fitz.Page", *, min_chars: int = settings.min_native_chars) -> PageKind:
    """Classify a single PDF page.

    Rules:
      * ``NATIVE``  — the page exposes extractable digital text whose stripped
        length is at least ``min_chars``.
      * ``SCANNED`` — the page has too little extractable text but contains one
        or more images (a scanned/photographed page that needs OCR).
      * ``EMPTY``   — neither meaningful text nor any image.

    Args:
        page: A ``fitz.Page`` instance from an open ``fitz.Document``.
        min_chars: Minimum stripped text length to treat the page as native.

    Returns:
        The :class:`PageKind` for this page.
    """
    try:
        text = page.get_text("text") or ""
    except Exception:  # pragma: no cover - corrupt page stream
        logger.warning("classify_page: failed to extract text from page; treating as empty")
        text = ""

    if len(text.strip()) >= min_chars:
        return PageKind.NATIVE

    try:
        has_images = bool(page.get_images(full=True))
    except Exception:  # pragma: no cover - corrupt page stream
        logger.warning("classify_page: failed to list images on page")
        has_images = False

    if has_images:
        return PageKind.SCANNED

    return PageKind.EMPTY


def classify_pdf(path: str | Path) -> list[PageKind]:
    """Classify every page of a PDF, preserving 1-based page order.

    Opens the document with ``fitz``, classifies each page, and always closes
    the document before returning.

    Args:
        path: Path to the PDF file.

    Returns:
        A list of :class:`PageKind`, one entry per page in document order.
    """
    pdf_path = Path(path)
    kinds: list[PageKind] = []

    doc = fitz.open(str(pdf_path))
    try:
        for page in doc:
            kinds.append(classify_page(page))
    finally:
        doc.close()

    native = sum(1 for k in kinds if k is PageKind.NATIVE)
    scanned = sum(1 for k in kinds if k is PageKind.SCANNED)
    empty = sum(1 for k in kinds if k is PageKind.EMPTY)
    logger.info(
        "classify_pdf %s: %d pages (native=%d scanned=%d empty=%d)",
        pdf_path.name,
        len(kinds),
        native,
        scanned,
        empty,
    )
    return kinds
