"""PDF text extraction and page rendering primitives.

Thin, well-typed wrappers around PyMuPDF (`fitz`) used by the ingestion
pipeline:

* :func:`open_pdf`           -- open a document (caller is responsible for
  closing it).
* :func:`extract_native_text` -- pull reading-order text out of a page that
  already contains a digital text layer.
* :func:`render_page_image`  -- rasterize a page to a Pillow ``Image`` for the
  OCR backends (used for scanned pages).

``fitz`` and ``PIL.Image`` are lightweight enough to import at module top
(see CONTRACT.md). Heavy OCR / ML libraries live in ``ingest.ocr`` and are
imported lazily there.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import fitz  # PyMuPDF; lightweight, allowed at module top.
from PIL import Image

from config import settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from fitz import Document, Page

logger = get_logger(__name__)


def open_pdf(path: str | Path) -> "fitz.Document":
    """Open *path* as a PDF and return the :class:`fitz.Document`.

    The caller owns the returned document and must close it (e.g. via a
    ``with`` block or ``doc.close()``). Raises :class:`FileNotFoundError` if
    the file does not exist and propagates ``fitz`` errors for malformed PDFs.
    """
    pdf_path = Path(path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if not pdf_path.is_file():
        raise FileNotFoundError(f"Not a file: {pdf_path}")

    logger.debug("Opening PDF: %s", pdf_path)
    doc = fitz.open(str(pdf_path))
    return doc


def extract_native_text(page: "Page") -> str:
    """Extract the digital text layer of *page* in natural reading order.

    Uses PyMuPDF's ``"text"`` extraction mode, which sorts blocks/lines into a
    human reading order and is the right choice for native (non-scanned) pages.
    Returns an empty string when the page has no extractable text (e.g. a
    scanned image page); the caller should route such pages to OCR.
    """
    try:
        text = page.get_text("text")
    except Exception:  # pragma: no cover - defensive; PyMuPDF edge cases
        logger.exception(
            "Native text extraction failed on page %s", _page_label(page)
        )
        return ""

    if not text:
        return ""

    # Normalise line endings; downstream normalization handles the rest.
    return text.replace("\r\n", "\n").replace("\r", "\n")


def render_page_image(page: "Page", dpi: int = settings.ocr_dpi) -> "Image.Image":
    """Rasterize *page* to an RGB :class:`PIL.Image.Image` at *dpi*.

    Intended for scanned pages that must be sent to an OCR backend. The page is
    rendered with PyMuPDF at the requested DPI; any alpha channel produced by
    ``get_pixmap`` is dropped so the result is always 3-channel RGB.
    """
    if dpi <= 0:
        raise ValueError(f"dpi must be positive, got {dpi}")

    pix = page.get_pixmap(dpi=dpi)

    # ``pix.n`` includes the alpha channel when present; for >3 components
    # (RGBA / CMYK+alpha) re-render onto an opaque RGB pixmap so PIL gets clean
    # 3-byte samples. Grayscale (n==1) is widened to RGB too.
    if pix.n != 3 or pix.alpha:
        pix = fitz.Pixmap(fitz.csRGB, pix)

    image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    logger.debug(
        "Rendered page %s at %d dpi -> %dx%d",
        _page_label(page),
        dpi,
        image.width,
        image.height,
    )
    return image


def _page_label(page: "Page") -> str:
    """Best-effort 1-based page label for log messages."""
    number = getattr(page, "number", None)
    if isinstance(number, int):
        return str(number + 1)
    return "?"
