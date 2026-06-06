"""Recover a book's chapter structure for the comprehension layer.

Given a PDF's table of contents (and, as a gated fallback, heading heuristics),
produce a flat list of top-level :class:`Section`s covering the whole book. The
ingestion pipeline then groups chunks under each section and summarizes them.

Pure-python and light to import: ``extract_toc`` only calls a method on a passed
``fitz`` document; nothing here imports a heavy library at module load.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from config import settings
from core.logging import get_logger
from core.models import Chunk

if TYPE_CHECKING:  # pragma: no cover - typing only
    import fitz

    from core.models import PageContent

logger = get_logger(__name__)


@dataclass
class Section:
    """A top-level structural unit of a book (a chapter), spanning a page range."""

    title: str
    level: int          # TOC depth (top level) or 0 for a synthesized whole-book section
    page_start: int     # 1-based, inclusive
    page_end: int       # 1-based, inclusive


def extract_toc(doc: "fitz.Document") -> list[tuple[int, str, int]]:
    """Return the PDF outline as ``(level, title, page)`` tuples (1-based pages).

    Returns ``[]`` when the document has no usable outline or PyMuPDF raises.
    """
    try:
        raw = doc.get_toc(simple=True) or []
    except Exception as exc:  # noqa: BLE001 - outline is best-effort
        logger.debug("get_toc failed: %s", exc)
        return []
    out: list[tuple[int, str, int]] = []
    for entry in raw:
        try:
            level, title, page = int(entry[0]), str(entry[1] or "").strip(), int(entry[2])
        except (TypeError, ValueError, IndexError):
            continue
        out.append((level, title, page))
    return out


def _sections_from_toc(
    toc: list[tuple[int, str, int]], num_pages: int
) -> list[Section]:
    """Build top-level chapter sections from a TOC.

    Uses only the shallowest TOC level (the chapters); deeper entries fold into
    their chapter. Each chapter spans from its page up to the page before the
    next chapter (the last runs to the end of the book).
    """
    entries = [(lvl, title, pg) for (lvl, title, pg) in toc if pg and pg >= 1]
    if not entries:
        return []
    top = min(lvl for lvl, _, _ in entries)
    chapters = sorted(
        [(title, pg) for (lvl, title, pg) in entries if lvl == top],
        key=lambda t: t[1],
    )
    sections: list[Section] = []
    for i, (title, pg) in enumerate(chapters):
        start = max(1, min(pg, num_pages))
        if i + 1 < len(chapters):
            end = max(start, min(chapters[i + 1][1] - 1, num_pages))
        else:
            end = max(start, num_pages)
        sections.append(
            Section(title=title or f"Section {i + 1}", level=top, page_start=start, page_end=end)
        )
    return sections


# Heading heuristics (gated fallback). Patterns match POST-normalization text
# (see ingest.normalize): Arabic alef/ya folded, tatweel stripped, diacritics
# usually preserved. Kept deliberately conservative.
_AR_HEADING = re.compile(
    r"^\s*(الفصل|الباب|المبحث|الجزء|القسم|المقدمة|الخاتمة|تمهيد|توطئة)\b"
)
_EN_HEADING = re.compile(
    r"^\s*(chapter|part|section|introduction|conclusion|preface|epilogue|prologue)\b",
    re.IGNORECASE,
)


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    if not line or len(line) > 80:        # headings are short
        return False
    return bool(_AR_HEADING.match(line) or _EN_HEADING.match(line))


def _detect_headings(pages: "list[PageContent]", num_pages: int) -> list[Section]:
    """Heuristic structure when there is no TOC: scan each page's opening lines.

    Treats a page whose first non-empty line looks like a heading as the start of
    a new section. Returns ``[]`` when fewer than two headings are found (not
    enough signal to beat a single whole-book section).
    """
    starts: list[tuple[str, int]] = []
    for page in pages:
        for line in (page.text or "").splitlines():
            line = line.strip()
            if not line:
                continue
            if _looks_like_heading(line):
                starts.append((line[:80], page.page_number))
            break  # only inspect the first non-empty line of each page
    if len(starts) < 2:
        return []
    starts.sort(key=lambda t: t[1])
    sections: list[Section] = []
    for i, (title, pg) in enumerate(starts):
        start = max(1, min(pg, num_pages))
        end = (
            max(start, min(starts[i + 1][1] - 1, num_pages))
            if i + 1 < len(starts)
            else max(start, num_pages)
        )
        sections.append(Section(title=title, level=1, page_start=start, page_end=end))
    return sections


def detect_structure(
    pages: "list[PageContent]",
    toc: list[tuple[int, str, int]],
    num_pages: int,
) -> list[Section]:
    """Return the book's chapter sections, always covering the whole book.

    TOC first (safe and accurate). If empty and heading detection is enabled,
    fall back to heading heuristics. As a last resort, a single whole-book
    section so the comprehension layer always has at least one unit to summarize.
    """
    sections = _sections_from_toc(toc, num_pages)
    if not sections and settings.enable_heading_detection:
        sections = _detect_headings(pages, num_pages)
        if sections:
            logger.info("Structure via heading heuristics: %d section(s)", len(sections))
    if not sections:
        return [Section(title="(whole book)", level=0, page_start=1, page_end=max(1, num_pages))]
    logger.info("Structure detected: %d chapter section(s)", len(sections))
    return sections


def _section_index_for_page(sections: list[Section], page: int) -> int:
    """Index of the section containing ``page`` (nearest by start if none does)."""
    for i, s in enumerate(sections):
        if s.page_start <= page <= s.page_end:
            return i
    return min(range(len(sections)), key=lambda i: abs(sections[i].page_start - page))


def assign_chunks_to_sections(
    chunks: list[Chunk], sections: list[Section]
) -> dict[int, list[Chunk]]:
    """Bucket each chunk under the section that contains its starting page."""
    buckets: dict[int, list[Chunk]] = {i: [] for i in range(len(sections))}
    if not sections:
        return buckets
    for ch in chunks:
        buckets[_section_index_for_page(sections, ch.page_start)].append(ch)
    return buckets
