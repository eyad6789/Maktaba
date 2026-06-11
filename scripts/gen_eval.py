"""Golden eval-set generator for the book RAG system.

Builds a retrieval-evaluation question set from the LIVE ingested corpus: it
scrolls every ``level=="passage"`` point out of Qdrant (never the PDFs), picks
a stratified sample of chunks per book, asks the LLM chain to write ONE
self-contained quiz question per chunk (Arabic or English, with a configurable
cross-language fraction), quality-filters and dedups the questions, and writes
two JSONL files:

* ``--out`` (default ``data/eval/questions.jsonl``) — the golden set consumed
  by ``scripts.eval``. One object per line::

      {"question": str, "expect_book_id": str, "expect_page": int,
       "expect_chunk_id": str, "lang": "ar"|"en", "qtype": "same"|"cross",
       "book_title": str, "source_pages": [int, int]}

* ``--review-out`` (default ``data/eval/questions_review.jsonl``) — the same
  objects plus ``"source_text"`` (the full chunk text) for human/LLM review;
  ``scripts.eval`` never reads it.

Usage
-----
    python -m scripts.gen_eval
    python -m scripts.gen_eval --per-book 12 --cross-lang-frac 0.25 --seed 13

Heavy backends (Qdrant, the LLM engine) are imported lazily so this module
stays importable and ``py_compile``-clean on a CPU-only box.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import settings
from core.logging import get_logger
from ingest.normalize import detect_lang, normalize_text

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from retrieval.search import QdrantStore

logger = get_logger(__name__)


# --- prompts ------------------------------------------------------------------

_SYSTEM = (
    "You write ONE self-contained, specific quiz question that is answerable "
    "ONLY from the given book excerpt, in the requested language (Arabic or "
    "English). The question must name a concrete entity, event, or claim from "
    "the excerpt so it stands alone. It must NOT reference the excerpt itself: "
    "never use phrases like 'the passage', 'this text', 'the excerpt', "
    "'النص', 'المقتطف', or 'الفقرة'. Output the question text alone — no "
    "numbering, no quotes, no preamble. "
    "اكتب سؤال اختبار واحدا محددا وقائما بذاته يمكن الاجابة عنه فقط من مقتطف "
    "الكتاب المعطى، باللغة المطلوبة. يجب ان يذكر السؤال كيانا او حدثا او ادعاء "
    "محددا ورد في المقتطف، والا يشير الى المقتطف نفسه. اخرج نص السؤال فقط دون "
    "ترقيم او اقتباس او مقدمات."
)

# Appended to the user message on the single quality retry.
_NUDGE = (
    "Your previous attempt was rejected. Write a NEW question: 10-200 "
    "characters, strictly in the requested language only, naming a concrete "
    "fact from the excerpt, and never mentioning the passage/text/excerpt "
    "(or النص/المقتطف/الفقرة)."
)

# --- quality filter vocabulary --------------------------------------------------

# Meta-phrases that make a question unanswerable without the excerpt in front
# of you. English phrases are matched casefolded; Arabic ones as whole words
# bounded by non-Arabic letters — a bare substring test would reject innocent
# words that merely contain "النص" (e.g. النصر, النصف, النصيحة).
_BANNED_EN = (
    "the passage",
    "this text",
    "the excerpt",
    "the author writes",
    "according to the text",
)
_BANNED_AR_RE = re.compile(
    r"(?<![ء-ي])(?:النص|المقتطف|الفقرة|هذا الكتاب يقول)(?![ء-ي])"
)

_MIN_QUESTION_CHARS = 10
_MAX_QUESTION_CHARS = 200

# Lines like "1. ...", "1) ...", "- ...", "* ..." — strip the list decoration.
_LIST_PREFIX = re.compile(r"^\s*(?:[-*•]|\d{1,2}[.)])\s*")
# Leading "Question:" / "السؤال:" labels some models prepend despite the prompt.
_LABEL_PREFIX = re.compile(r"^(?:question|السؤال)\s*[::]\s*", re.IGNORECASE)

# Script counters for the majority-script fallback in resolve_chunk_lang.
_ARABIC_RE = re.compile(r"[؀-ۿ]")
_LATIN_RE = re.compile(r"[A-Za-z]")


# --- corpus access --------------------------------------------------------------


def fetch_passages(
    store: "QdrantStore", *, page_size: int = 256
) -> dict[str, list[dict[str, Any]]]:
    """Scroll every passage-level point out of Qdrant, grouped by book.

    Paginates ``store.client.scroll`` with a ``level == "passage"`` payload
    filter until exhausted. Each returned chunk dict carries: ``chunk_id``
    (the point id as str), ``text``, ``book_id``, ``title``, ``author``,
    ``page_start``, ``page_end``, ``lang``, ``chunk_index``.
    """
    from qdrant_client import models as qm  # lazy

    scroll_filter = qm.Filter(
        must=[qm.FieldCondition(key="level", match=qm.MatchValue(value="passage"))]
    )

    by_book: dict[str, list[dict[str, Any]]] = {}
    offset: Any = None
    while True:
        points, offset = store.client.scroll(
            collection_name=store.collection,
            scroll_filter=scroll_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in points:
            payload = point.payload or {}
            chunk = {
                "chunk_id": str(point.id),
                "text": payload.get("text", ""),
                "book_id": str(payload.get("book_id", "")),
                "title": payload.get("title", ""),
                "author": payload.get("author"),
                "page_start": int(payload.get("page_start", 0)),
                "page_end": int(payload.get("page_end", 0)),
                "lang": payload.get("lang") or "unknown",
                "chunk_index": int(payload.get("chunk_index", 0)),
            }
            by_book.setdefault(chunk["book_id"], []).append(chunk)
        if offset is None:
            break

    logger.info(
        "Fetched %d passage(s) across %d book(s) from %r",
        sum(len(v) for v in by_book.values()),
        len(by_book),
        store.collection,
    )
    return by_book


# --- pure helpers (unit-tested) --------------------------------------------------


def sample_chunks(
    by_book: dict[str, list[dict[str, Any]]],
    *,
    per_book: int,
    seed: int,
    min_chars: int,
) -> list[dict[str, Any]]:
    """Pick a stratified, deterministic sample of chunks from each book.

    Pure function. Per book: chunks shorter than ``min_chars`` are dropped,
    the rest are sorted by ``chunk_index`` and split into ``per_book`` evenly
    spaced strata; the middle chunk of each stratum is picked, with a ±1
    tie-jitter from ``random.Random(seed)`` so repeated runs with a different
    seed don't always probe the exact same sentence. Small books yield fewer
    than ``per_book`` chunks. Output order is stable: ``(title, chunk_index)``.
    """
    rng = random.Random(seed)
    picked: list[dict[str, Any]] = []

    def _book_key(book_id: str) -> tuple[str, str]:
        chunks = by_book[book_id]
        title = (chunks[0].get("title") or "") if chunks else ""
        return (title, book_id)

    for book_id in sorted(by_book, key=_book_key):
        eligible = [
            c for c in by_book[book_id] if len(c.get("text") or "") >= min_chars
        ]
        eligible.sort(key=lambda c: c["chunk_index"])
        n = len(eligible)
        if n == 0:
            continue
        k = min(per_book, n)
        for i in range(k):
            # Disjoint stratum [lo, hi]; pick its middle, jittered by ±1.
            lo = (i * n) // k
            hi = max(lo, ((i + 1) * n) // k - 1)
            mid = (lo + hi) // 2
            if hi > lo:
                mid = min(hi, max(lo, mid + rng.choice((-1, 0, 1))))
            picked.append(eligible[mid])

    picked.sort(
        key=lambda c: (c.get("title") or "", c["chunk_index"], c["book_id"])
    )
    return picked


def resolve_chunk_lang(chunk_lang: str | None, text: str) -> str:
    """Resolve a chunk's payload language to a binary ``"ar"`` | ``"en"``.

    Pure function. ``"ar"``/``"en"`` pass through; ``"mixed"``/``"unknown"``/
    missing are re-detected from ``text`` via :func:`detect_lang`, falling back
    to the majority script (Arabic vs Latin letter count) when detection stays
    ambiguous. Letter-free text resolves to ``"en"``.
    """
    if chunk_lang in ("ar", "en"):
        return chunk_lang
    detected = detect_lang(text or "")
    if detected in ("ar", "en"):
        return detected
    arabic = len(_ARABIC_RE.findall(text or ""))
    latin = len(_LATIN_RE.findall(text or ""))
    return "ar" if arabic > latin else "en"


def pick_question_lang(chunk_lang: str, i: int, cross_frac: float) -> str:
    """Choose the question language for sample index ``i``.

    Pure function. ``chunk_lang`` must already be resolved to ``"ar"``/``"en"``
    (see :func:`resolve_chunk_lang`; anything else is treated as ``"en"``).
    With ``cross_frac > 0``, every ``round(1/cross_frac)``-th sample (those
    where ``i % period == 0``, so i=0,4,8,... at 0.25) gets the OTHER language;
    the rest — and everything when ``cross_frac <= 0`` — keep the chunk's own.
    """
    base = "ar" if chunk_lang == "ar" else "en"
    if cross_frac <= 0:
        return base
    other = "en" if base == "ar" else "ar"
    period = max(1, round(1 / cross_frac))
    return other if i % period == 0 else base


def quality_ok(question: str, chunk: dict[str, Any], target_lang: str) -> bool:
    """Decide whether a generated question is usable as a golden eval item.

    Pure function. Rejects questions that are empty, shorter than 10 or longer
    than 200 characters, contain a banned meta-phrase (casefolded English /
    Arabic whole-word), or whose language disagrees with ``target_lang``.
    A ``"mixed"`` detection is accepted when the question's majority script
    matches the target — cross-language questions legitimately keep proper
    nouns in the other script (e.g. an Arabic question citing "Darwin").
    ``chunk`` is accepted for signature stability (future checks may ground
    the question against its source text).
    """
    del chunk  # reserved for future source-grounding checks
    q = (question or "").strip()
    if not q or len(q) < _MIN_QUESTION_CHARS or len(q) > _MAX_QUESTION_CHARS:
        return False
    folded = q.casefold()
    if any(phrase in folded for phrase in _BANNED_EN):
        return False
    if _BANNED_AR_RE.search(q):
        return False
    detected = detect_lang(q)
    if detected == target_lang:
        return True
    if detected == "mixed":
        arabic = len(_ARABIC_RE.findall(q))
        latin = len(_LATIN_RE.findall(q))
        return ("ar" if arabic > latin else "en") == target_lang
    return False


def build_record(
    chunk: dict[str, Any], question: str, target_lang: str
) -> dict[str, Any]:
    """Build one golden-set JSONL record (the contract ``scripts.eval`` reads).

    Pure function. ``expect_page`` is the chunk's ``page_start``; ``qtype`` is
    ``"same"`` when the question language matches the chunk's resolved
    language, ``"cross"`` otherwise.
    """
    chunk_lang = resolve_chunk_lang(chunk.get("lang"), chunk.get("text", ""))
    return {
        "question": question,
        "expect_book_id": str(chunk["book_id"]),
        "expect_page": int(chunk["page_start"]),
        "expect_chunk_id": str(chunk["chunk_id"]),
        "lang": target_lang,
        "qtype": "same" if target_lang == chunk_lang else "cross",
        "book_title": chunk.get("title", ""),
        "source_pages": [int(chunk["page_start"]), int(chunk["page_end"])],
    }


def _clean_question(text: str) -> str | None:
    """First non-empty line of LLM output, stripped of list/label decoration."""
    for raw in (text or "").splitlines():
        line = _LIST_PREFIX.sub("", raw).strip().strip('"“”').strip()
        line = _LABEL_PREFIX.sub("", line).strip()
        if line:
            return line
    return None


# --- LLM call ---------------------------------------------------------------------


def generate_question(
    chunk: dict[str, Any],
    target_lang: str,
    *,
    nudge: str | None = None,
    retry_delay: float = 2.0,
) -> str | None:
    """Ask the LLM for one quiz question about ``chunk`` in ``target_lang``.

    One completion via :mod:`llm.engine` (mirroring ``retrieval.expand``:
    through the cloud fallback chain when configured, otherwise pinned to the
    local utility model). Retries once after ``retry_delay`` seconds on
    provider exceptions — callers running under a rate limiter should pass
    their throttle interval so the retry never bursts past the budget.
    Returns the cleaned question text, or ``None`` on failure.
    """
    from llm import engine  # lazy

    # Same chain/utility-model selection as retrieval.expand: the chain
    # (Gemini-fast) when configured and available, else the local utility model.
    use_chain = settings.expansion_use_chain and settings.llm_backend == "fallback"
    model = None if use_chain else settings.utility_model

    lang_name = "Arabic" if target_lang == "ar" else "English"
    user = (
        f"Book title: {chunk.get('title') or 'Unknown'}\n"
        f"Author: {chunk.get('author') or 'Unknown'}\n"
        f"Requested language: {lang_name}\n\n"
        f"Excerpt:\n{(chunk.get('text') or '').strip()}\n\n"
        f"Write the one quiz question in {lang_name}:"
    )
    if nudge:
        user += "\n\n" + nudge

    for attempt in range(2):
        try:
            out = engine.complete(
                _SYSTEM,
                [{"role": "user", "content": user}],
                model=model,
                max_tokens=120,
                temperature=0.3,
            )
            return _clean_question(out)
        except Exception as exc:  # noqa: BLE001 - one retry, then skip the chunk
            logger.warning(
                "Question generation failed for chunk %s (%s)%s",
                chunk.get("chunk_id"),
                exc,
                "; retrying" if attempt == 0 else "",
            )
            if attempt == 0:
                time.sleep(retry_delay)
    return None


# --- generation run ----------------------------------------------------------------


class _Throttle:
    """Sleep between LLM calls so the run stays under ``rpm`` calls/minute."""

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._last = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        now = time.monotonic()
        remaining = self._last + self.interval - now
        if remaining > 0:
            time.sleep(remaining)
        self._last = time.monotonic()


def _generate_all(
    sampled: list[dict[str, Any]], *, cross_frac: float, rpm: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Generate, filter, and dedup one question per sampled chunk.

    Returns ``(records, review_records, per_book_stats)`` where stats map
    ``book_id -> {"title", "sampled", "kept", "skipped"}``.
    """
    throttle = _Throttle(rpm)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Any]] = {}

    for i, chunk in enumerate(sampled):
        book_stats = stats.setdefault(
            chunk["book_id"],
            {"title": chunk.get("title", ""), "sampled": 0, "kept": 0, "skipped": 0},
        )
        book_stats["sampled"] += 1

        base_lang = resolve_chunk_lang(chunk.get("lang"), chunk.get("text", ""))
        target_lang = pick_question_lang(base_lang, i, cross_frac)

        question: str | None = None
        for nudge in (None, _NUDGE):  # one corrective retry on quality failure
            throttle.wait()
            candidate = generate_question(
                chunk,
                target_lang,
                nudge=nudge,
                retry_delay=max(2.0, throttle.interval),
            )
            if candidate is None:
                continue
            if not quality_ok(candidate, chunk, target_lang):
                logger.info(
                    "Rejected question for chunk %s: %r", chunk["chunk_id"], candidate
                )
                continue
            key = normalize_text(candidate).casefold()
            if key in seen:
                logger.info(
                    "Duplicate question for chunk %s: %r", chunk["chunk_id"], candidate
                )
                continue
            seen.add(key)
            question = candidate
            break

        if question is None:
            logger.warning(
                "Skipping chunk %s (%s p.%s): no acceptable question",
                chunk["chunk_id"],
                chunk.get("title"),
                chunk.get("page_start"),
            )
            book_stats["skipped"] += 1
            continue

        record = build_record(chunk, question, target_lang)
        records.append(record)
        reviews.append({**record, "source_text": chunk.get("text", "")})
        book_stats["kept"] += 1

    return records, reviews, stats


# --- output ------------------------------------------------------------------------


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """Write ``rows`` as UTF-8 JSONL, creating parent directories as needed."""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _print_summary(
    stats: dict[str, dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    out: str,
    review_out: str,
) -> None:
    """Print per-book and aggregate counts for the run."""
    print("=" * 72)
    totals = Counter()
    for book_id, s in sorted(stats.items(), key=lambda kv: (kv[1]["title"], kv[0])):
        print(
            f"{s['title'] or book_id}: "
            f"sampled={s['sampled']} kept={s['kept']} skipped={s['skipped']}"
        )
        totals.update(
            sampled=s["sampled"], kept=s["kept"], skipped=s["skipped"]
        )
    print("-" * 72)
    print(
        f"Total: sampled={totals['sampled']} "
        f"kept={totals['kept']} skipped={totals['skipped']}"
    )
    lang_counts = Counter(r["lang"] for r in records)
    qtype_counts = Counter(r["qtype"] for r in records)
    print("By lang:  " + ", ".join(f"{k}={v}" for k, v in sorted(lang_counts.items())))
    print("By qtype: " + ", ".join(f"{k}={v}" for k, v in sorted(qtype_counts.items())))
    print(f"Wrote {len(records)} question(s) -> {out}")
    print(f"Review copy (adds source_text) -> {review_out}")


# --- CLI ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.gen_eval",
        description="Generate a golden eval question set from the ingested Qdrant corpus.",
    )
    parser.add_argument(
        "--out",
        default="data/eval/questions.jsonl",
        help="Golden-set JSONL output path (default: %(default)s).",
    )
    parser.add_argument(
        "--review-out",
        default="data/eval/questions_review.jsonl",
        help="Review JSONL output path, adds source_text (default: %(default)s).",
    )
    parser.add_argument(
        "--per-book",
        type=int,
        default=8,
        help="Chunks sampled per book (default: %(default)s).",
    )
    parser.add_argument(
        "--cross-lang-frac",
        type=float,
        default=0.25,
        help="Fraction of questions asked in the OTHER language (default: %(default)s).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=13,
        help="Sampling seed for deterministic chunk picks (default: %(default)s).",
    )
    parser.add_argument(
        "--rpm",
        type=int,
        default=8,
        help="Max LLM calls per minute (default: %(default)s).",
    )
    parser.add_argument(
        "--min-chars",
        type=int,
        default=200,
        help="Minimum chunk text length to be sampled (default: %(default)s).",
    )
    parser.add_argument(
        "--collection",
        default=settings.qdrant_collection,
        help="Qdrant collection to read passages from (default: %(default)s).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = _parse_args(argv)

    if args.per_book <= 0:
        print(f"--per-book must be positive (got {args.per_book}).", file=sys.stderr)
        return 2
    if not 0.0 <= args.cross_lang_frac <= 1.0:
        print(
            f"--cross-lang-frac must be in [0, 1] (got {args.cross_lang_frac}).",
            file=sys.stderr,
        )
        return 2
    if args.rpm <= 0:
        print(f"--rpm must be positive (got {args.rpm}).", file=sys.stderr)
        return 2
    if args.min_chars < 0:
        print(f"--min-chars must be >= 0 (got {args.min_chars}).", file=sys.stderr)
        return 2

    # Lazy heavy import — only when we actually run a generation.
    from retrieval.search import QdrantStore

    store = QdrantStore(collection=args.collection)
    by_book = fetch_passages(store)
    total_passages = sum(len(v) for v in by_book.values())
    if total_passages == 0:
        print(
            f"No passages found in collection {args.collection!r}.", file=sys.stderr
        )
        return 1
    print(
        f"Fetched {total_passages} passage(s) across {len(by_book)} book(s) "
        f"from {args.collection!r}."
    )

    sampled = sample_chunks(
        by_book, per_book=args.per_book, seed=args.seed, min_chars=args.min_chars
    )
    if not sampled:
        print("No chunks passed the --min-chars filter.", file=sys.stderr)
        return 1
    print(f"Sampled {len(sampled)} chunk(s); generating questions (rpm={args.rpm})...")

    records, reviews, stats = _generate_all(
        sampled, cross_frac=args.cross_lang_frac, rpm=args.rpm
    )
    if not records:
        print("No questions survived quality filtering.", file=sys.stderr)
        return 1

    _write_jsonl(args.out, records)
    _write_jsonl(args.review_out, reviews)
    _print_summary(stats, records, out=args.out, review_out=args.review_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
