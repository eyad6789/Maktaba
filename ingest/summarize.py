"""Build chapter- and book-level summary nodes (the comprehension layer).

For each detected :class:`~ingest.structure.Section`, summarize its chunks into a
``chapter_summary`` node; then reduce the chapter summaries into one
``book_summary`` node. The nodes are ordinary :class:`~core.models.Chunk`s
(tagged by ``level``), so they ride the existing embed + upsert path unchanged
and become retrievable alongside raw passages.

Long sections are handled map/reduce: chunk texts are batched to
``summary_map_batch_tokens``, each batch summarized, then the partials reduced
into the chapter brief — keeping every LLM call within a CPU model's context.

The LLM is reached only through the lazy ``llm.engine.complete`` entrypoint, so
this module imports cleanly without a model installed (tests fake the engine).
"""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from config import settings
from core.logging import get_logger
from core.models import BookMeta, Chunk
from ingest.chunk import approx_tokens
from ingest.structure import Section, assign_chunks_to_sections

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass

logger = get_logger(__name__)

# chunk_index range reserved for summary nodes so they never collide with the
# 0..N passage indices of a book.
_CHAPTER_INDEX_BASE = 1_000_000
_BOOK_INDEX = 2_000_000


def _summary_id(book_id: str, suffix: str) -> str:
    """Deterministic id for a summary node (stable across re-ingests)."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{book_id}:summary:{suffix}"))


def _summarize_one(
    text: str, *, kind: str, lang: str | None, title: str | None, model: str | None
) -> str:
    """Run a single summarization LLM call and return the text."""
    from llm import engine  # lazy: keeps this module importable without a model
    from llm.prompts import build_summarize_prompt, summarize_system_prompt

    # An explicit model pins the local server (the fallback chain's rule). When
    # summary_use_chain is on and the chain is the active backend, pass None so
    # summaries ride the cloud chain (Gemini/Claude) — orders of magnitude
    # faster than a CPU model, and the chain still ends locally without keys.
    if model is None and settings.summary_use_chain and settings.llm_backend == "fallback":
        effective_model = None
    else:
        effective_model = model or settings.summary_model

    system = summarize_system_prompt(kind)
    user = build_summarize_prompt(text, kind=kind, lang=lang, title=title)
    out = engine.complete(
        system,
        [{"role": "user", "content": user}],
        model=effective_model,
        max_tokens=settings.summary_max_tokens,
        temperature=0.3,
    )
    return (out or "").strip()


def _batch_texts(texts: list[str], max_tokens: int) -> list[list[str]]:
    """Group texts so each batch's approx token total stays under ``max_tokens``."""
    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for t in texts:
        tok = approx_tokens(t)
        if current and current_tokens + tok > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append(t)
        current_tokens += tok
    if current:
        batches.append(current)
    return batches


def _summarize_long(
    texts: list[str], *, kind: str, lang: str | None, title: str | None, model: str | None
) -> str:
    """Summarize possibly-long text via map/reduce; single call when it fits."""
    combined = "\n\n".join(t for t in texts if t.strip())
    if approx_tokens(combined) <= settings.summary_map_batch_tokens:
        return _summarize_one(combined, kind=kind, lang=lang, title=title, model=model)

    batches = _batch_texts(texts, settings.summary_map_batch_tokens)
    logger.debug("Map step: %d batch(es) for %r", len(batches), title)
    partials = [
        _summarize_one("\n\n".join(b), kind="partial", lang=lang, title=title, model=model)
        for b in batches
    ]
    reduced_input = "\n\n".join(p for p in partials if p.strip())
    return _summarize_one(reduced_input, kind=kind, lang=lang, title=title, model=model)


def _node_lang(book: BookMeta, chunks: list[Chunk]) -> str | None:
    """Pick the language to write a summary in: book language, else majority chunk."""
    if book.language and book.language != "mixed":
        return book.language
    counts: dict[str, int] = {}
    for c in chunks:
        if c.lang:
            counts[c.lang] = counts.get(c.lang, 0) + 1
    return max(counts, key=counts.get) if counts else book.language


def summarize_section(
    section: Section,
    chunks: list[Chunk],
    book: BookMeta,
    *,
    index: int,
    parent_id: str,
    model: str | None = None,
) -> Chunk:
    """Summarize one section's chunks into a ``chapter_summary`` node."""
    lang = _node_lang(book, chunks)
    text = _summarize_long(
        [c.text for c in chunks], kind="chapter", lang=lang, title=section.title, model=model
    )
    return Chunk(
        chunk_id=_summary_id(book.book_id, f"chapter:{index}"),
        book_id=book.book_id,
        title=book.title,
        author=book.author,
        text=text,
        page_start=section.page_start,
        page_end=section.page_end,
        chunk_index=_CHAPTER_INDEX_BASE + index,
        lang=lang,
        token_count=approx_tokens(text),
        level="chapter_summary",
        parent_id=parent_id,
        chapter_title=section.title,
    )


def _make_book_node(
    book: BookMeta, node_id: str, source_text: str, *, lang: str | None, model: str | None
) -> Chunk:
    text = _summarize_one(source_text, kind="book", lang=lang, title=book.title, model=model)
    page_end = max(1, book.num_pages)
    return Chunk(
        chunk_id=node_id,
        book_id=book.book_id,
        title=book.title,
        author=book.author,
        text=text,
        page_start=1,
        page_end=page_end,
        chunk_index=_BOOK_INDEX,
        lang=lang,
        token_count=approx_tokens(text),
        level="book_summary",
        parent_id=None,
        chapter_title=None,
    )


def build_summary_nodes(
    sections: list[Section],
    chunks: list[Chunk],
    book: BookMeta,
    *,
    model: str | None = None,
) -> list[Chunk]:
    """Build chapter + book summary nodes for one book.

    With multiple sections: one ``chapter_summary`` per (non-tiny) section plus a
    ``book_summary`` reduced from them. With a single whole-book section: just a
    ``book_summary`` reduced from the chunks (a chapter node would duplicate it).
    Returns ``[]`` when there are no chunks. Deterministic ids make re-ingest
    idempotent (Qdrant upsert overwrites).
    """
    if not chunks:
        return []

    book_node_id = _summary_id(book.book_id, "book")
    chapter_nodes: list[Chunk] = []

    if len(sections) > 1:
        buckets = assign_chunks_to_sections(chunks, sections)
        for i, section in enumerate(sections):
            sec_chunks = buckets.get(i, [])
            if not sec_chunks:
                continue
            sec_tokens = sum(c.token_count for c in sec_chunks)
            if sec_tokens < settings.summary_min_section_tokens:
                logger.debug("Skipping tiny section %r (%d tok)", section.title, sec_tokens)
                continue
            chapter_nodes.append(
                summarize_section(
                    section, sec_chunks, book, index=i, parent_id=book_node_id, model=model
                )
            )

    book_lang = _node_lang(book, chunks)
    if chapter_nodes:
        book_source = "\n\n".join(
            f"{(n.chapter_title or 'Chapter')}: {n.text}" for n in chapter_nodes
        )
    else:
        book_source = "\n\n".join(c.text for c in chunks if c.text.strip())

    book_node = _make_book_node(
        book, book_node_id, book_source, lang=book_lang, model=model
    )

    nodes = chapter_nodes + [book_node]
    logger.info(
        "Built %d summary node(s) for %s (%d chapter + 1 book)",
        len(nodes),
        book.book_id,
        len(chapter_nodes),
    )
    return nodes
