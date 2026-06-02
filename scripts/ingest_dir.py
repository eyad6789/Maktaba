"""CLI: ingest every PDF under a directory.

Usage
-----
Enqueue all books found under ``<dir>`` onto the ingestion queue (default)::

    python -m scripts.ingest_dir data/books

Run ingestion inline in this process (no Redis/RQ worker required)::

    python -m scripts.ingest_dir data/books --sync

Limit the walk to the top-level directory only::

    python -m scripts.ingest_dir data/books --no-recursive

By default the directory is walked recursively for ``*.pdf`` files; pass
``--no-recursive`` to scan only the given directory. In ``--sync`` mode the
shared singletons (Qdrant store, embedder, registry, OCR backend) are built
once and :func:`ingest.pipeline.ingest_book` is called for each file, printing
an :class:`~core.models.IngestStats` summary per book. Otherwise each file is
enqueued via :func:`ingest.worker.enqueue_book` and its job id is printed.

Heavy libraries (FlagEmbedding, OCR backends, qdrant_client, RQ) are imported
lazily by the collaborating modules, so importing this script and running it in
the default (enqueue) mode stays cheap.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from config import settings
from core.logging import get_logger

logger = get_logger(__name__)


def find_pdfs(directory: Path, *, recursive: bool = True) -> list[Path]:
    """Return a sorted list of ``*.pdf`` files under ``directory``.

    Matching is case-insensitive on the suffix so ``.PDF`` files are included.
    When ``recursive`` is true the whole tree is walked, otherwise only the
    immediate children of ``directory`` are scanned.
    """
    globber = directory.rglob if recursive else directory.glob
    pdfs = [
        path
        for path in globber("*")
        if path.is_file() and path.suffix.lower() == ".pdf"
    ]
    return sorted(set(pdfs))


def _ingest_sync(pdfs: list[Path]) -> int:
    """Ingest each PDF inline, printing per-book stats. Return exit code."""
    # Lazy imports: these pull in heavy backends (torch/transformers/qdrant).
    from ingest.embed import BGEM3Embedder
    from ingest.ocr import get_ocr_backend
    from ingest.pipeline import ingest_book
    from ingest.registry import Registry
    from retrieval.search import QdrantStore

    store = QdrantStore()
    store.ensure_collection()
    embedder = BGEM3Embedder()
    registry = Registry()

    # OCR is only needed for scanned pages; build it best-effort so native
    # books still ingest if the backend (or its model) is unavailable here.
    ocr = None
    try:
        ocr = get_ocr_backend(settings.ocr_backend)
    except Exception as exc:  # noqa: BLE001 - backend init is best-effort
        logger.warning(
            "Could not initialize OCR backend %r (%s); "
            "scanned pages will be skipped",
            settings.ocr_backend,
            exc,
        )

    had_failure = False
    for index, pdf in enumerate(pdfs, start=1):
        print(f"[{index}/{len(pdfs)}] ingesting {pdf} ...")
        stats = ingest_book(
            pdf,
            store=store,
            embedder=embedder,
            registry=registry,
            ocr=ocr,
        )
        if stats.status == "failed":
            had_failure = True
        print(stats.model_dump_json(indent=2))

    return 1 if had_failure else 0


def _ingest_enqueue(pdfs: list[Path]) -> int:
    """Enqueue each PDF for asynchronous ingestion. Return exit code."""
    from ingest.worker import enqueue_book

    for index, pdf in enumerate(pdfs, start=1):
        job_id = enqueue_book(str(pdf))
        print(f"[{index}/{len(pdfs)}] enqueued {pdf} -> job {job_id}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse args, discover PDFs, and ingest them (inline or via the queue)."""
    parser = argparse.ArgumentParser(
        prog="ingest_dir",
        description="Ingest every PDF under a directory into the RAG store.",
    )
    parser.add_argument(
        "dir",
        help="Directory to scan for *.pdf files.",
    )
    parser.add_argument(
        "--recursive",
        dest="recursive",
        action="store_true",
        default=True,
        help="Recurse into subdirectories (default).",
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Scan only the top-level directory.",
    )
    parser.add_argument(
        "--sync",
        action="store_true",
        help="Run ingestion inline instead of enqueuing jobs.",
    )
    args = parser.parse_args(argv)

    directory = Path(args.dir).expanduser()
    if not directory.exists():
        print(f"error: directory not found: {directory}", file=sys.stderr)
        return 2
    if not directory.is_dir():
        print(f"error: not a directory: {directory}", file=sys.stderr)
        return 2

    pdfs = find_pdfs(directory, recursive=args.recursive)
    if not pdfs:
        print(f"No PDF files found under {directory}")
        return 0

    mode = "sync" if args.sync else "enqueue"
    print(
        f"Found {len(pdfs)} PDF file(s) under {directory} "
        f"(recursive={args.recursive}); mode={mode}"
    )

    if args.sync:
        return _ingest_sync(pdfs)
    return _ingest_enqueue(pdfs)


if __name__ == "__main__":
    raise SystemExit(main())
