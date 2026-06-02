"""Benchmark ingestion + query throughput and extrapolate to the full corpus.

Run this on the GPU server against a *representative* sample of books (e.g. 10,
ideally mixing native-text and scanned PDFs) to estimate how long ingesting the
whole library (1000+ books x ~400 pages) will take, and to measure query
latency.

Usage
-----
    # ingestion timing + projection to 1000 books of 400 pages, 30% scanned, 4 workers
    python -m scripts.benchmark --books data/sample --total-books 1000 \
        --pages-per-book 400 --scanned-frac 0.30 --workers 4

    # also benchmark query latency over a question set
    python -m scripts.benchmark --books data/sample --queries data/questions.jsonl --k 8

Notes
-----
* Ingestion timing runs the real pipeline inline (``--sync`` semantics): native
  pages are fast, scanned pages are OCR-bound and dominate runtime, so a mixed
  sample gives the most honest projection. Books already in the registry are
  skipped by the pipeline and reported as such (re-hash a fresh copy or clear
  the registry to re-time them).
* Heavy backends are imported lazily inside :func:`main`; the pure projection
  math (:func:`extrapolate`) and timing core (:func:`time_ingestion`) accept
  injected collaborators so they are unit-testable without a GPU.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ingest.embed import BGEM3Embedder
    from ingest.ocr import OCRBackend
    from ingest.registry import Registry
    from retrieval.rerank import Reranker
    from retrieval.search import QdrantStore

logger = get_logger(__name__)

SECONDS_PER_HOUR = 3600.0


# --- data shapes ------------------------------------------------------------


@dataclass
class BookTiming:
    """Timing + composition for one ingested book."""

    path: str
    seconds: float
    num_pages: int
    native_pages: int
    scanned_pages: int
    num_chunks: int
    status: str

    @property
    def sec_per_page(self) -> float:
        return self.seconds / self.num_pages if self.num_pages else 0.0


@dataclass
class Projection:
    """Extrapolated full-corpus ingestion estimate."""

    total_books: int
    pages_per_book: int
    scanned_frac: float
    workers: int
    native_rate: float | None          # sec/page on native pages (None if unknown)
    scanned_rate: float | None         # sec/page on scanned pages (None if unknown)
    overall_rate: float                # sec/page across all sampled pages
    est_seconds_single: float          # one worker
    est_seconds_parallel: float        # `workers` workers

    @property
    def est_hours_parallel(self) -> float:
        return self.est_seconds_parallel / SECONDS_PER_HOUR


@dataclass
class QueryTiming:
    """Aggregate query-latency stats (seconds)."""

    count: int
    mean: float
    p50: float
    p95: float
    samples: list[float] = field(default_factory=list)


# --- ingestion timing -------------------------------------------------------


def time_ingestion(
    pdfs: list[Path],
    *,
    store: "QdrantStore",
    embedder: "BGEM3Embedder",
    registry: "Registry",
    ocr: "OCRBackend | None" = None,
) -> list[BookTiming]:
    """Ingest each PDF inline, timing the full pipeline per book.

    Returns one :class:`BookTiming` per file (in input order).
    """
    from ingest.pipeline import ingest_book  # lazy: pulls pipeline graph

    timings: list[BookTiming] = []
    for index, pdf in enumerate(pdfs, start=1):
        logger.info("[%d/%d] timing ingestion of %s", index, len(pdfs), pdf)
        start = time.perf_counter()
        stats = ingest_book(
            pdf, store=store, embedder=embedder, registry=registry, ocr=ocr
        )
        elapsed = time.perf_counter() - start
        timings.append(
            BookTiming(
                path=str(pdf),
                seconds=elapsed,
                num_pages=stats.num_pages,
                native_pages=stats.native_pages,
                scanned_pages=stats.scanned_pages,
                num_chunks=stats.num_chunks,
                status=stats.status,
            )
        )
        logger.info(
            "  %.2fs (%d pages: %d native, %d scanned -> %d chunks) [%s]",
            elapsed,
            stats.num_pages,
            stats.native_pages,
            stats.scanned_pages,
            stats.num_chunks,
            stats.status,
        )
    return timings


def extrapolate(
    timings: list[BookTiming],
    *,
    total_books: int,
    pages_per_book: int,
    scanned_frac: float,
    workers: int = 1,
) -> Projection:
    """Project full-corpus ingestion time from sampled per-book timings.

    Native and scanned per-page rates are estimated separately (OCR is far
    slower), using books that are purely native / purely scanned. When a pure
    sample is unavailable for a class, the overall sampled rate is used as a
    fallback for that class.
    """
    timed = [t for t in timings if t.status in {"completed", "failed"} and t.num_pages]
    total_pages = sum(t.num_pages for t in timed)
    total_seconds = sum(t.seconds for t in timed)
    overall_rate = (total_seconds / total_pages) if total_pages else 0.0

    def _rate(predicate, page_attr: str) -> float | None:
        books = [t for t in timed if predicate(t)]
        pages = sum(getattr(t, page_attr) for t in books)
        secs = sum(t.seconds for t in books)
        return (secs / pages) if pages else None

    native_rate = _rate(lambda t: t.scanned_pages == 0 and t.native_pages > 0, "native_pages")
    scanned_rate = _rate(lambda t: t.native_pages == 0 and t.scanned_pages > 0, "scanned_pages")

    eff_native = native_rate if native_rate is not None else overall_rate
    eff_scanned = scanned_rate if scanned_rate is not None else overall_rate

    scanned_frac = min(max(scanned_frac, 0.0), 1.0)
    proj_total_pages = total_books * pages_per_book
    proj_scanned_pages = proj_total_pages * scanned_frac
    proj_native_pages = proj_total_pages - proj_scanned_pages

    est_single = proj_native_pages * eff_native + proj_scanned_pages * eff_scanned
    workers = max(1, workers)
    est_parallel = est_single / workers

    return Projection(
        total_books=total_books,
        pages_per_book=pages_per_book,
        scanned_frac=scanned_frac,
        workers=workers,
        native_rate=native_rate,
        scanned_rate=scanned_rate,
        overall_rate=overall_rate,
        est_seconds_single=est_single,
        est_seconds_parallel=est_parallel,
    )


# --- query latency ----------------------------------------------------------


def _percentile(sorted_vals: list[float], pct: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_vals:
        return 0.0
    rank = max(0, min(len(sorted_vals) - 1, int(round(pct / 100.0 * (len(sorted_vals) - 1)))))
    return sorted_vals[rank]


def time_queries(
    questions: list[str],
    *,
    embedder: "BGEM3Embedder",
    store: "QdrantStore",
    reranker: "Reranker",
    k: int = settings.rerank_top_k,
) -> QueryTiming:
    """Time the embed -> hybrid-search -> rerank path for each question."""
    samples: list[float] = []
    for q in questions:
        start = time.perf_counter()
        emb = embedder.embed_query(q)
        candidates = store.hybrid_search(emb, top_k=settings.search_top_k)
        reranker.rerank(q, candidates, top_k=k)
        samples.append(time.perf_counter() - start)

    ordered = sorted(samples)
    mean = (sum(samples) / len(samples)) if samples else 0.0
    return QueryTiming(
        count=len(samples),
        mean=mean,
        p50=_percentile(ordered, 50),
        p95=_percentile(ordered, 95),
        samples=samples,
    )


# --- reporting --------------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < SECONDS_PER_HOUR:
        return f"{seconds / 60:.1f}m"
    if seconds < 24 * SECONDS_PER_HOUR:
        return f"{seconds / SECONDS_PER_HOUR:.1f}h"
    return f"{seconds / (24 * SECONDS_PER_HOUR):.1f}d"


def print_report(timings: list[BookTiming], projection: Projection) -> None:
    print("\n=== Ingestion timing (sample) ===")
    print(f"{'book':<40} {'pages':>6} {'nat':>5} {'scan':>5} {'chunks':>7} {'sec':>8} {'s/pg':>7}")
    print("-" * 86)
    for t in timings:
        name = Path(t.path).name
        name = name if len(name) <= 38 else name[:37] + "…"
        print(
            f"{name:<40} {t.num_pages:>6} {t.native_pages:>5} {t.scanned_pages:>5} "
            f"{t.num_chunks:>7} {t.seconds:>8.2f} {t.sec_per_page:>7.3f}"
        )

    total_pages = sum(t.num_pages for t in timings)
    total_sec = sum(t.seconds for t in timings)
    total_chunks = sum(t.num_chunks for t in timings)
    print("-" * 86)
    print(
        f"{'TOTAL':<40} {total_pages:>6} "
        f"{sum(t.native_pages for t in timings):>5} "
        f"{sum(t.scanned_pages for t in timings):>5} "
        f"{total_chunks:>7} {total_sec:>8.2f} "
        f"{(total_sec / total_pages if total_pages else 0):>7.3f}"
    )

    p = projection
    print("\n=== Per-page rates ===")
    print(f"  native : {f'{p.native_rate:.3f} s/page' if p.native_rate is not None else 'n/a (no pure-native sample)'}")
    print(f"  scanned: {f'{p.scanned_rate:.3f} s/page' if p.scanned_rate is not None else 'n/a (no pure-scanned sample)'}")
    print(f"  overall: {p.overall_rate:.3f} s/page")

    print(f"\n=== Projection -> {p.total_books} books x {p.pages_per_book} pages "
          f"({p.scanned_frac:.0%} scanned) ===")
    print(f"  est. total pages : {p.total_books * p.pages_per_book:,}")
    print(f"  1 worker         : {_fmt_duration(p.est_seconds_single)}")
    print(f"  {p.workers} worker(s)        : {_fmt_duration(p.est_seconds_parallel)} "
          f"({p.est_hours_parallel:.1f}h)")
    if total_chunks and total_pages:
        est_chunks = (total_chunks / total_pages) * p.total_books * p.pages_per_book
        print(f"  est. total chunks: ~{est_chunks:,.0f} (vector DB size)")


def print_query_report(qt: QueryTiming) -> None:
    print("\n=== Query latency (embed + hybrid search + rerank) ===")
    print(f"  queries: {qt.count}")
    print(f"  mean   : {qt.mean * 1000:.0f} ms")
    print(f"  p50    : {qt.p50 * 1000:.0f} ms")
    print(f"  p95    : {qt.p95 * 1000:.0f} ms")


# --- CLI --------------------------------------------------------------------


def _find_pdfs(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() == ".pdf")


def _load_questions(path: Path) -> list[str]:
    out: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                obj: Any = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = obj.get("question") if isinstance(obj, dict) else None
            if isinstance(q, str) and q.strip():
                out.append(q.strip())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.benchmark",
        description="Benchmark ingestion/query throughput and project to the full corpus.",
    )
    parser.add_argument("--books", required=True, help="Directory of sample PDFs to time.")
    parser.add_argument("--total-books", type=int, default=1000, help="Corpus size to project to.")
    parser.add_argument("--pages-per-book", type=int, default=400, help="Avg pages/book to project.")
    parser.add_argument("--scanned-frac", type=float, default=0.30, help="Expected fraction of scanned pages.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel ingest workers to assume.")
    parser.add_argument("--queries", help="Optional JSONL of questions to benchmark query latency.")
    parser.add_argument("--k", type=int, default=settings.rerank_top_k, help="Rerank top-k for query timing.")
    args = parser.parse_args(argv)

    books_dir = Path(args.books).expanduser()
    if not books_dir.is_dir():
        print(f"error: --books is not a directory: {books_dir}", file=sys.stderr)
        return 2
    pdfs = _find_pdfs(books_dir)
    if not pdfs:
        print(f"error: no PDFs under {books_dir}", file=sys.stderr)
        return 2

    # Lazy heavy imports — only when actually benchmarking.
    from ingest.embed import BGEM3Embedder
    from ingest.ocr import get_ocr_backend
    from ingest.registry import Registry
    from retrieval.search import QdrantStore

    store = QdrantStore()
    store.ensure_collection()
    embedder = BGEM3Embedder()
    registry = Registry()
    try:
        ocr = get_ocr_backend(settings.ocr_backend)
    except Exception as exc:  # noqa: BLE001 - OCR optional for native-only samples
        logger.warning("OCR backend unavailable (%s); scanned pages will be skipped", exc)
        ocr = None

    print(f"Benchmarking ingestion of {len(pdfs)} sample book(s) from {books_dir} ...")
    timings = time_ingestion(pdfs, store=store, embedder=embedder, registry=registry, ocr=ocr)
    projection = extrapolate(
        timings,
        total_books=args.total_books,
        pages_per_book=args.pages_per_book,
        scanned_frac=args.scanned_frac,
        workers=args.workers,
    )
    print_report(timings, projection)

    if args.queries:
        qpath = Path(args.queries).expanduser()
        questions = _load_questions(qpath) if qpath.is_file() else []
        if questions:
            from retrieval.rerank import Reranker

            reranker = Reranker()
            qt = time_queries(questions, embedder=embedder, store=store, reranker=reranker, k=args.k)
            print_query_report(qt)
        else:
            print(f"\n(no questions loaded from {qpath}; skipping query benchmark)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
