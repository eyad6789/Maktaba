"""RQ-based ingestion worker.

Enqueues and runs single-book ingestion jobs on a Redis Queue (RQ) so the
GPU-bound, long-running ingestion pipeline executes out-of-band from the API.

Usage
-----
Enqueue from the API/CLI::

    from ingest.worker import enqueue_book
    job_id = enqueue_book("/data/books/foo.pdf", title="Foo", author="Bar")

Run a worker that consumes the queue::

    python -m ingest.worker

Heavy/optional libraries (``redis``, ``rq``, model backends) are imported
lazily inside the functions so this module stays importable on a CPU-only box
and passes ``py_compile``.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from config import settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import rq
    import redis

logger = get_logger(__name__)

# Generous default: a large scanned book OCR'd page-by-page can take hours.
_JOB_TIMEOUT = "6h"
# Keep finished job results around long enough for callers to poll them.
_RESULT_TTL = 86400  # seconds (24h)
# Keep failed jobs around for inspection/requeue.
_FAILURE_TTL = 604800  # seconds (7d)
# Minimum seconds between job.meta progress saves (each one is a Redis trip).
_PROGRESS_SAVE_INTERVAL = 1.0


def get_redis_connection() -> "redis.Redis":
    """Open a Redis connection from ``settings.redis_url``."""
    import redis  # lazy: optional dep, not needed for py_compile/import

    return redis.Redis.from_url(settings.redis_url)


def get_queue() -> "rq.Queue":
    """Return the RQ queue bound to ``settings.ingest_queue`` on Redis."""
    import rq  # lazy

    connection = get_redis_connection()
    return rq.Queue(settings.ingest_queue, connection=connection)


def enqueue_book(path: str, **kw: Any) -> str:
    """Enqueue a single book for asynchronous ingestion.

    Parameters
    ----------
    path:
        Absolute path to the PDF on the worker's filesystem.
    **kw:
        Forwarded to :func:`ingest_book_job` (e.g. ``title``, ``author``).

    Returns
    -------
    str
        The RQ job id, usable to poll job status/result later.
    """
    queue = get_queue()
    # Seed meta at enqueue time so the dashboard can label the job while it is
    # still queued; the worker enriches it (book_id, stage) once it starts.
    from pathlib import Path

    meta = {"path": path, "title": kw.get("title") or Path(path).stem}
    # NB: pass args/kwargs explicitly (not positionally). RQ's enqueue() asserts
    # no positional args are mixed with an explicit `kwargs=`/`args=` keyword.
    job = queue.enqueue(
        ingest_book_job,
        args=(path,),
        kwargs=dict(kw),
        job_timeout=_JOB_TIMEOUT,
        result_ttl=_RESULT_TTL,
        failure_ttl=_FAILURE_TTL,
        meta=meta,
    )
    logger.info("Enqueued ingest job %s for %s", job.id, path)
    return job.id


def _make_progress_callback(job: "rq.job.Job") -> Callable[[str, int, int], None]:
    """Build an ``on_progress`` callback that mirrors progress into ``job.meta``.

    Saves are throttled to one Redis round-trip per second (the terminal
    ``"done"`` stage is always saved so the dashboard sees completion). All
    Redis work is wrapped: a hiccup degrades to stale progress, never to a
    failed ingestion job.
    """
    last_save = 0.0
    last_stage: str | None = None

    def on_progress(stage: str, current: int, total: int) -> None:
        nonlocal last_save, last_stage
        now = time.monotonic()
        # Stage transitions always save (they can fire in quick succession but
        # are few); only same-stage ticks (per-page extraction) are throttled.
        if (
            stage == last_stage
            and stage != "done"
            and now - last_save < _PROGRESS_SAVE_INTERVAL
        ):
            return
        last_stage = stage
        try:
            job.meta["stage"] = stage
            job.meta["current"] = current
            job.meta["total"] = total
            job.save_meta()
            last_save = now
        except Exception as exc:  # noqa: BLE001 - progress is advisory only
            logger.debug("Could not save job progress (%s %d/%d): %s",
                         stage, current, total, exc)

    return on_progress


def ingest_book_job(path: str, **kw: Any) -> dict:
    """Worker entrypoint: ingest one book end-to-end.

    Constructs the shared singletons (vector store, embedder, registry, OCR
    backend), runs the ingestion pipeline, and returns the resulting
    :class:`~core.models.IngestStats` as a plain ``dict`` (RQ-serializable).

    Accepted keyword arguments
    ---------------------------
    title, author:
        Optional metadata overrides passed through to the pipeline.
    ocr_backend:
        Optional OCR backend name override (defaults to ``settings.ocr_backend``).

    Returns
    -------
    dict
        ``IngestStats.model_dump()``.
    """
    # Lazy imports: these pull in heavy backends (torch/transformers/qdrant).
    from ingest.embed import BGEM3Embedder
    from ingest.ocr import get_ocr_backend
    from ingest.pipeline import derive_book_id, ingest_book
    from ingest.registry import Registry
    from retrieval.search import QdrantStore

    title = kw.pop("title", None)
    author = kw.pop("author", None)
    force = bool(kw.pop("force", False))
    ocr_backend_name = kw.pop("ocr_backend", settings.ocr_backend)
    if kw:
        logger.warning("Ignoring unexpected ingest kwargs: %s", sorted(kw))

    logger.info("Starting ingest job for %s", path)

    # When running under RQ, expose identifying metadata + live progress on the
    # job so the dashboard can poll it. get_current_job() is None when this
    # function is called synchronously (tests, direct invocation) — everything
    # below then degrades to a no-op. All meta work is best-effort: a Redis
    # hiccup must never kill the ingestion itself.
    job = None
    try:
        import rq  # lazy: optional dep, absent on a box without the queue

        job = rq.get_current_job()
    except Exception:  # noqa: BLE001 - no rq/redis: run without job metadata
        job = None

    on_progress: Callable[[str, int, int], None] | None = None
    if job is not None:
        try:
            job.meta["path"] = path
            job.meta["title"] = title or Path(path).stem
            # book_id is deterministic from the file hash, so the dashboard can
            # link job -> book before the pipeline has even started.
            job.meta["book_id"] = derive_book_id(Registry.compute_hash(path))
            job.save_meta()
        except Exception as exc:  # noqa: BLE001 - meta is advisory only
            logger.warning("Could not record job metadata for %s: %s", path, exc)
        on_progress = _make_progress_callback(job)

    store = QdrantStore()
    store.ensure_collection()

    embedder = BGEM3Embedder()
    registry = Registry()

    # OCR is only needed for scanned pages; if the backend cannot be built
    # (e.g. missing model on this host), proceed without it so native books
    # still ingest. The pipeline records per-page OCR failures itself.
    ocr = None
    try:
        ocr = get_ocr_backend(ocr_backend_name)
    except Exception as exc:  # noqa: BLE001 - backend init is best-effort
        logger.warning(
            "Could not initialize OCR backend %r (%s); "
            "scanned pages will be skipped",
            ocr_backend_name,
            exc,
        )

    stats = ingest_book(
        path,
        store=store,
        embedder=embedder,
        registry=registry,
        ocr=ocr,
        title=title,
        author=author,
        force=force,
        on_progress=on_progress,
    )
    logger.info(
        "Finished ingest job for %s: status=%s chunks=%d",
        path,
        stats.status,
        stats.num_chunks,
    )
    return stats.model_dump()


def run_worker() -> None:
    """Start an RQ worker consuming :func:`get_queue` until interrupted."""
    import rq  # lazy

    connection = get_redis_connection()
    queue = rq.Queue(settings.ingest_queue, connection=connection)
    worker = rq.Worker([queue], connection=connection)
    logger.info(
        "Starting RQ worker on queue %r (redis=%s)",
        settings.ingest_queue,
        settings.redis_url,
    )
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    run_worker()
