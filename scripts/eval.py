"""Retrieval-quality evaluation CLI for the book RAG system.

Reads a JSONL file of questions (one JSON object per line), runs the full
online retrieval path for each — BGE-M3 query embedding -> Qdrant hybrid search
-> cross-encoder rerank — and reports ``recall@k``: the fraction of questions
whose expected source was retrieved in the top ``k`` reranked results.

Each input line may contain::

    {"question": "...",            # required
     "expect_book_id": "<uuid>",   # optional gold book id
     "expect_page": 42,            # optional gold page (1-based)
     "book_ids": ["<uuid>", ...]}  # optional retrieval filter

A question counts as a *hit* when:

* it has no expectations (``expect_book_id`` / ``expect_page``) — treated as a
  smoke-test hit if anything was retrieved; or
* one of the top-``k`` reranked results is from ``expect_book_id`` (when given),
  and — if ``expect_page`` is given — that result's page span
  ``[page_start, page_end]`` contains the expected page.

Usage
-----
    python -m scripts.eval questions.jsonl
    python -m scripts.eval questions.jsonl --k 8
    python -m scripts.eval questions.jsonl --answer   # also print Claude answer

Heavy backends (embedder, reranker, Qdrant, Anthropic) are imported lazily so
this module stays importable and ``py_compile``-clean on a CPU-only box.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import settings
from core.logging import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from core.models import SearchResult

logger = get_logger(__name__)


# --- input parsing ----------------------------------------------------------


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    """Parse a JSONL eval file into a list of question dicts.

    Blank lines are skipped. Each non-blank line must be a JSON object with a
    non-empty ``question`` string; malformed or question-less lines are logged
    and skipped rather than aborting the whole run.
    """
    questions: list[dict[str, Any]] = []
    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("Skipping malformed JSON on line %d: %s", lineno, exc)
                continue
            if not isinstance(obj, dict):
                logger.warning("Skipping non-object on line %d", lineno)
                continue
            question = obj.get("question")
            if not isinstance(question, str) or not question.strip():
                logger.warning("Skipping line %d: missing/empty 'question'", lineno)
                continue
            questions.append(obj)
    return questions


# --- hit logic --------------------------------------------------------------


def _result_matches(
    result: "SearchResult",
    expect_book_id: str | None,
    expect_page: int | None,
) -> bool:
    """Return True if one retrieved result satisfies the expectations."""
    if expect_book_id is not None and result.book_id != expect_book_id:
        return False
    if expect_page is not None:
        # page span is inclusive; tolerate unset/zero spans gracefully.
        if not (result.page_start <= expect_page <= result.page_end):
            return False
    return True


def is_hit(
    results: list["SearchResult"],
    expect_book_id: str | None,
    expect_page: int | None,
) -> bool:
    """Decide whether ``results`` (already truncated to top-k) is a hit.

    With no expectations, a hit just means something was retrieved (smoke test).
    Otherwise at least one result must match the book id and (if given) page.
    """
    if expect_book_id is None and expect_page is None:
        return len(results) > 0
    return any(_result_matches(r, expect_book_id, expect_page) for r in results)


# --- evaluation -------------------------------------------------------------


def evaluate(
    questions: list[dict[str, Any]],
    *,
    k: int,
    show_answer: bool = False,
) -> float:
    """Run retrieval (+ optional answering) for each question; print results.

    Returns the overall recall@k as a float in ``[0, 1]`` (0.0 when there are
    no scoreable questions).
    """
    # Lazy heavy imports — only when we actually run an evaluation.
    from ingest.embed import BGEM3Embedder
    from retrieval.rerank import Reranker
    from retrieval.search import QdrantStore

    embedder = BGEM3Embedder()
    store = QdrantStore()
    reranker = Reranker()

    total = len(questions)
    scoreable = 0
    hits = 0

    print(f"Evaluating {total} question(s) | recall@{k}")
    print("=" * 72)

    for idx, item in enumerate(questions, start=1):
        question = str(item["question"]).strip()
        expect_book_id = item.get("expect_book_id")
        expect_page = item.get("expect_page")
        book_ids = item.get("book_ids")

        if expect_book_id is not None:
            expect_book_id = str(expect_book_id)
        if expect_page is not None:
            try:
                expect_page = int(expect_page)
            except (TypeError, ValueError):
                logger.warning(
                    "Q%d: invalid 'expect_page' %r; ignoring", idx, expect_page
                )
                expect_page = None
        if book_ids is not None and not isinstance(book_ids, list):
            logger.warning("Q%d: 'book_ids' is not a list; ignoring", idx)
            book_ids = None

        try:
            query_emb = embedder.embed_query(question)
            candidates = store.hybrid_search(
                query_emb,
                top_k=settings.search_top_k,
                book_ids=[str(b) for b in book_ids] if book_ids else None,
            )
            reranked = reranker.rerank(question, candidates, top_k=k)
        except Exception as exc:  # noqa: BLE001 - report and continue the run
            logger.error("Q%d failed during retrieval: %s", idx, exc)
            print(f"[Q{idx}] ERROR: {exc}")
            print("-" * 72)
            continue

        top = reranked[:k]
        has_expectation = expect_book_id is not None or expect_page is not None
        hit = is_hit(top, expect_book_id, expect_page)

        if has_expectation:
            scoreable += 1
            if hit:
                hits += 1

        _print_question_result(
            idx=idx,
            question=question,
            expect_book_id=expect_book_id,
            expect_page=expect_page,
            has_expectation=has_expectation,
            hit=hit,
            results=top,
        )

        if show_answer:
            _print_answer(question, top, item.get("model"))

        print("-" * 72)

    recall = (hits / scoreable) if scoreable else 0.0
    print("=" * 72)
    if scoreable:
        print(
            f"recall@{k} = {recall:.3f} ({hits}/{scoreable} scoreable questions)"
        )
    else:
        print(
            f"No questions had expectations; ran {total} retrieval smoke test(s)."
        )
    if scoreable != total:
        print(f"({total - scoreable} question(s) had no expectations and were not scored.)")
    return recall


def _print_question_result(
    *,
    idx: int,
    question: str,
    expect_book_id: str | None,
    expect_page: int | None,
    has_expectation: bool,
    hit: bool,
    results: list["SearchResult"],
) -> None:
    """Pretty-print one question's outcome and its top retrieved sources."""
    if has_expectation:
        marker = "HIT " if hit else "MISS"
    else:
        marker = "----"
    print(f"[Q{idx}] [{marker}] {question}")

    expect_bits: list[str] = []
    if expect_book_id is not None:
        expect_bits.append(f"book_id={expect_book_id}")
    if expect_page is not None:
        expect_bits.append(f"page={expect_page}")
    if expect_bits:
        print("       expect: " + ", ".join(expect_bits))

    if not results:
        print("       (no results retrieved)")
        return

    for rank, r in enumerate(results, start=1):
        score = r.rerank_score if r.rerank_score is not None else r.score
        print(
            f"       {rank:>2}. score={score:.4f} "
            f"book={r.book_id} pages={r.page_start}-{r.page_end} "
            f'"{_snippet(r.title)}"'
        )


def _print_answer(
    question: str, results: list["SearchResult"], model: str | None
) -> None:
    """Generate and print a grounded Claude answer for ``question``.

    Best-effort: failures (missing key, network, optional module) are logged
    and reported inline so the recall metric is never lost.
    """
    try:
        from llm.answer import answer_question

        answer = answer_question(question, results, model=model)
    except Exception as exc:  # noqa: BLE001 - answering is optional
        logger.warning("Answer generation failed: %s", exc)
        print(f"       answer: <unavailable: {exc}>")
        return

    grounded = "grounded" if answer.grounded else "ungrounded"
    print(f"       answer ({answer.model}, {grounded}):")
    for line in answer.answer.splitlines() or [""]:
        print(f"         {line}")
    if answer.citations:
        cites = "; ".join(
            f"{c.title} p.{c.page_start}-{c.page_end}" for c in answer.citations
        )
        print(f"       citations: {cites}")


def _snippet(text: str, limit: int = 60) -> str:
    """Single-line, length-capped preview of ``text`` for console output."""
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


# --- CLI --------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.eval",
        description="Evaluate retrieval recall@k over a JSONL question set.",
    )
    parser.add_argument(
        "questions",
        help="Path to a JSONL file: one {\"question\": ..., expect_*} object per line.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=settings.rerank_top_k,
        help="Number of reranked results to keep per question (default: %(default)s).",
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Also generate and print the grounded Claude answer per question.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns a process exit code."""
    args = _parse_args(argv)

    if args.k <= 0:
        print(f"--k must be a positive integer (got {args.k}).", file=sys.stderr)
        return 2

    path = Path(args.questions)
    if not path.is_file():
        print(f"Questions file not found: {path}", file=sys.stderr)
        return 2

    questions = load_questions(path)
    if not questions:
        print(f"No valid questions found in {path}.", file=sys.stderr)
        return 1

    evaluate(questions, k=args.k, show_answer=args.answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
