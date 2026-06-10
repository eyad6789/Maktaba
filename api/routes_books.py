"""Dashboard book-management endpoints: upload, job polling, listing, deletion.

Mounted onto the main FastAPI app (see ``api/main.py``), sharing its lifespan
singletons via ``request.app.state``:

* ``POST   /upload``          — multipart PDF upload, streamed to
  ``settings.uploads_dir`` with a size cap, deduped by content hash, then
  enqueued onto the RQ ingest queue.
* ``GET    /jobs/{job_id}``   — poll one ingestion job's state + live progress
  (mirrored into ``job.meta`` by ``ingest/worker.py``).
* ``GET    /jobs``            — recent jobs across queued/started/finished/
  failed, so the dashboard survives a page reload.
* ``GET    /books``           — registry rows plus corpus totals.
* ``DELETE /books/{book_id}`` — remove a book's vectors, registry row and
  (when it lives under the uploads dir) its source file.

Redis/RQ are imported lazily inside the handlers so this module stays
importable (and ``py_compile``-clean) on a box without the queue installed.
Response payloads are plain dicts — these are dashboard-internal shapes, not
part of the modelled query/chat API surface.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile

from config import settings
from core.logging import get_logger
from ingest.pipeline import derive_book_id  # light: heavy deps are lazy inside
from ingest.registry import STATUS_STARTED

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    import rq

    from ingest.registry import Registry
    from retrieval.search import QdrantStore

logger = get_logger(__name__)

router = APIRouter()

# Uploads are streamed to disk in chunks of this size; the cap is enforced
# incrementally so an oversized body never lands fully on disk.
_UPLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB

# Only PDFs are ingestible.
_PDF_SUFFIX = ".pdf"

# Most recent jobs returned by GET /jobs.
_MAX_JOBS = 50

# Filename sanitizer: keep Unicode word characters (so Arabic titles survive)
# plus dot and dash; every other run of characters collapses to "_".
_UNSAFE_CHARS = re.compile(r"[^\w.-]+", re.UNICODE)


# -- state accessors (mirror api/main.py) -------------------------------------


def _get_store(request: Request) -> "QdrantStore":
    store = getattr(request.app.state, "store", None)
    if store is None:  # pragma: no cover - defensive; lifespan always sets it
        raise HTTPException(status_code=503, detail="Vector store not initialized")
    return store


def _get_registry(request: Request) -> "Registry":
    registry = getattr(request.app.state, "registry", None)
    if registry is None:  # pragma: no cover - defensive
        raise HTTPException(status_code=503, detail="Registry not initialized")
    return registry


# -- helpers --------------------------------------------------------------------


def _sanitize_filename(name: str) -> str:
    """Reduce an arbitrary client filename to a safe ``*.pdf`` basename."""
    stem = Path(name).stem
    safe = _UNSAFE_CHARS.sub("_", stem).strip("._-") or "book"
    return f"{safe}{_PDF_SUFFIX}"


def _job_payload(job: "rq.job.Job") -> dict:
    """Flatten an RQ job into the dashboard's polling payload."""
    try:
        job.refresh()  # pull the latest meta written by the worker
    except Exception as exc:  # noqa: BLE001 - stale meta beats a 500
        logger.debug("Could not refresh job %s: %s", job.id, exc)
    meta = job.meta or {}
    # rq returns a JobStatus enum; normalize to its plain value ("queued"...).
    status = job.get_status(refresh=False)
    state = getattr(status, "value", None) or str(status or "unknown")
    error = None
    if state == "failed" and job.exc_info:
        # Last non-empty traceback line carries the exception type + message.
        lines = [ln for ln in str(job.exc_info).strip().splitlines() if ln.strip()]
        error = lines[-1] if lines else None
    return {
        "job_id": job.id,
        "state": state,
        "stage": meta.get("stage"),
        "current": meta.get("current"),
        "total": meta.get("total"),
        "book_id": meta.get("book_id"),
        "title": meta.get("title"),
        "path": meta.get("path"),
        "error": error,
        "result": job.result if state == "finished" else None,
    }


def _not_found_payload(job_id: str) -> dict:
    """Payload for an unknown/expired job id (same shape as `_job_payload`)."""
    return {
        "job_id": job_id,
        "state": "not_found",
        "stage": None,
        "current": None,
        "total": None,
        "book_id": None,
        "title": None,
        "path": None,
        "error": None,
        "result": None,
    }


# -- endpoints ------------------------------------------------------------------


@router.post("/upload")
def upload(
    request: Request,
    file: UploadFile = File(...),
    title: str | None = Form(None),
    author: str | None = Form(None),
) -> dict:
    """Accept a PDF upload from the browser and enqueue it for ingestion.

    Streams the body to ``settings.uploads_dir`` in 1 MiB chunks, rejecting
    non-PDF filenames (400) and bodies over ``settings.max_upload_mb`` (413).
    The saved file is named ``<hash8>_<sanitized>.pdf`` from its sha256, and
    already-ingested content (matched by hash) is reported as a duplicate
    without re-enqueueing.
    """
    registry = _get_registry(request)

    original = Path(file.filename or "").name
    if not original.lower().endswith(_PDF_SUFFIX):
        raise HTTPException(status_code=400, detail="Only .pdf files are accepted")

    uploads_dir = settings.uploads_dir
    uploads_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = settings.max_upload_mb * 1024 * 1024

    tmp_fd, tmp_name = tempfile.mkstemp(dir=uploads_dir, suffix=".part")
    tmp_path = Path(tmp_name)
    size_bytes = 0
    try:
        with os.fdopen(tmp_fd, "wb") as out:
            while True:
                chunk = file.file.read(_UPLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                size_bytes += len(chunk)
                if size_bytes > max_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"File exceeds the {settings.max_upload_mb} MB "
                            "upload limit"
                        ),
                    )
                out.write(chunk)
    except BaseException:
        # Any failure (size cap, client disconnect, disk error): no partials.
        tmp_path.unlink(missing_ok=True)
        raise

    file_hash = registry.compute_hash(tmp_path)
    book_id = derive_book_id(file_hash)
    filename = f"{file_hash[:8]}_{_sanitize_filename(original)}"

    if registry.is_ingested(file_hash):
        tmp_path.unlink(missing_ok=True)
        logger.info("Upload %r already ingested (book_id=%s)", original, book_id)
        return {
            "status": "duplicate",
            "book_id": book_id,
            "job_id": None,
            "filename": filename,
        }

    dest = uploads_dir / filename
    tmp_path.replace(dest)  # atomic; same name on re-upload just overwrites
    logger.info("Saved upload %r -> %s (%d bytes)", original, dest, size_bytes)

    # Lazy import keeps Redis/RQ off the import path of this module.
    from ingest.worker import enqueue_book

    try:
        job_id = enqueue_book(str(dest), title=title, author=author)
    except Exception as exc:  # noqa: BLE001 - queue may be unreachable
        logger.error("Failed to enqueue %s: %s", dest, exc)
        raise HTTPException(
            status_code=503,
            detail=f"Could not enqueue ingestion job: {exc}",
        ) from exc

    return {
        "status": "queued",
        "job_id": job_id,
        "book_id": book_id,
        "filename": filename,
        "size_bytes": size_bytes,
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    """Poll one ingestion job's state and live progress.

    Unknown ids return ``state="not_found"`` with HTTP 200 rather than 404:
    the dashboard polls this endpoint, and a job expiring out of Redis is an
    expected outcome, not an error.
    """
    import rq  # lazy
    from rq.exceptions import NoSuchJobError

    from ingest.worker import get_redis_connection

    try:
        job = rq.job.Job.fetch(job_id, connection=get_redis_connection())
    except NoSuchJobError:
        return _not_found_payload(job_id)
    except Exception as exc:  # noqa: BLE001 - Redis unreachable
        raise HTTPException(
            status_code=503, detail=f"Could not reach the job queue: {exc}"
        ) from exc
    return _job_payload(job)


@router.get("/jobs")
def list_jobs() -> dict:
    """List recent ingestion jobs across every lifecycle state.

    Union of the queue itself (queued) and RQ's Started/Finished/Failed
    registries, de-duplicated and capped at the 50 most recently enqueued —
    enough for the dashboard to rebuild its progress view after a reload.
    """
    import rq  # lazy
    from rq.registry import (
        FailedJobRegistry,
        FinishedJobRegistry,
        StartedJobRegistry,
    )

    from ingest.worker import get_queue

    try:
        queue = get_queue()
        connection = queue.connection
        job_ids: list[str] = list(queue.get_job_ids())
        for registry_cls in (StartedJobRegistry, FinishedJobRegistry, FailedJobRegistry):
            job_ids.extend(
                registry_cls(queue.name, connection=connection).get_job_ids()
            )
    except Exception as exc:  # noqa: BLE001 - Redis unreachable
        raise HTTPException(
            status_code=503, detail=f"Could not reach the job queue: {exc}"
        ) from exc

    seen: set[str] = set()
    jobs: list["rq.job.Job"] = []
    for job_id in job_ids:
        if job_id in seen:
            continue
        seen.add(job_id)
        try:
            jobs.append(rq.job.Job.fetch(job_id, connection=connection))
        except Exception:  # noqa: BLE001 - job expired between listing and fetch
            continue

    jobs.sort(
        key=lambda j: j.enqueued_at.timestamp() if j.enqueued_at else 0.0,
        reverse=True,
    )
    return {"jobs": [_job_payload(job) for job in jobs[:_MAX_JOBS]]}


@router.get("/books")
def list_books(request: Request) -> dict:
    """List every registry row plus corpus totals for the dashboard."""
    registry = _get_registry(request)
    store = _get_store(request)
    books = registry.list_books()
    return {
        "books": books,
        "total_books": len(books),
        "total_chunks": store.count(),
    }


@router.delete("/books/{book_id}")
def delete_book(request: Request, book_id: str) -> dict:
    """Delete one book: its vectors, its registry row and its uploaded file.

    Refuses with 409 while the book is mid-ingestion. The source file is
    unlinked only when it resolves under ``settings.uploads_dir`` — books
    ingested from arbitrary library paths are never touched on disk. Vector
    deletion runs first; if Qdrant fails, the registry row survives so the
    delete can be retried.
    """
    registry = _get_registry(request)
    store = _get_store(request)

    row = registry.get_by_book_id(book_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Unknown book_id: {book_id}")
    if row.get("status") == STATUS_STARTED:
        raise HTTPException(
            status_code=409,
            detail="Book is currently being ingested; try again when it finishes",
        )

    store.delete_by_book(book_id)
    registry.delete_by_book_id(book_id)

    source_path = row.get("source_path")
    if source_path:
        try:
            resolved = Path(source_path).resolve()
            if resolved.is_relative_to(settings.uploads_dir.resolve()) and resolved.is_file():
                resolved.unlink()
                logger.info("Removed uploaded source file %s", resolved)
        except Exception as exc:  # noqa: BLE001 - file removal is best-effort
            logger.warning("Could not remove source file %s: %s", source_path, exc)

    return {"deleted": True, "book_id": book_id}
