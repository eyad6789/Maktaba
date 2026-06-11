"""Retrieval-quality evaluation CLI for the book RAG system.

Reads a JSONL file of questions (one JSON object per line) and runs each
through the REAL online retrieval path — route classification
(:func:`retrieval.route.classify_route`) plus the full pipeline
(:func:`retrieval.pipeline.retrieve`: multi-query fusion, routed hybrid
search, rerank, score floor, small-to-big context expansion) — reporting:

* ``recall@k``  — fraction of questions whose gold source ranks in the top k
  (lenient tier: gold book id + page contained in the result's page span);
* ``MRR@k``     — mean reciprocal rank, truncated at k (see
  :func:`compute_metrics` for the truncation contract);
* ``nDCG@k``    — single-gold nDCG (IDCG = 1);
* ``strict@k``  — recall@k on the exact gold *chunk id* (strict tier, only
  for questions that carry ``expect_chunk_id``);
* ``mean-res``  — mean number of results returned per question, a
  prompt-noise proxy for the score-floor calibration work.

Metrics are sliced overall, by question language (``ar``/``en``) and by
``qtype`` (``same``-language vs ``cross``-language questions).

Multiple ABLATION configs run in ONE process (the heavy models load once);
each config temporarily overrides ``config.settings`` knobs through
:func:`override_settings`. LLM query expansion is memoized to
``--expansion-cache`` so every config — and every rerun — retrieves with
identical query variants.

Each input line may contain (the old minimal format still parses)::

    {"question": "...",              # required
     "expect_book_id": "<uuid>",     # gold book id
     "expect_page": 42,              # gold page (source chunk page_start)
     "expect_chunk_id": "<uuid>",    # gold chunk id -> strict tier
     "lang": "ar",                   # question language ("ar" | "en")
     "qtype": "same",                # question vs chunk language ("same"|"cross")
     "book_ids": ["<uuid>", ...]}    # optional retrieval filter

Extra keys (``book_title``, ``source_pages``, ...) are ignored. A question
with no expectations is a retrieval smoke test: marked hit when anything came
back, excluded from the metric aggregates.

Usage
-----
    python -m scripts.eval data/eval/questions.jsonl
    python -m scripts.eval data/eval/questions.jsonl --k 8 --ablations full,no-floor,legacy
    python -m scripts.eval data/eval/questions.jsonl --set rerank_min_score=0.25
    python -m scripts.eval data/eval/questions.jsonl --json report.json --dump-scores scores.json
    python -m scripts.eval data/eval/questions.jsonl --answer   # answer on last config

Heavy backends (embedder, reranker, Qdrant, Anthropic) are imported lazily so
this module stays importable and ``py_compile``-clean on a CPU-only box.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import sys
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from config import settings
from core.logging import get_logger
from ingest.normalize import detect_lang

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from core.models import SearchResult
    from ingest.embed import BGEM3Embedder
    from retrieval.rerank import Reranker
    from retrieval.search import QdrantStore

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


def _question_spec(item: dict[str, Any], idx: int) -> dict[str, Any]:
    """Normalize one JSONL line into typed gold expectations.

    Old-format lines (question + optional ``expect_book_id`` / ``expect_page``
    / ``book_ids``) parse fine: missing keys become ``None``/defaults; extra
    keys (``book_title``, ``source_pages``, ...) are ignored. ``lang`` falls
    back to :func:`ingest.normalize.detect_lang` on the question text and
    ``qtype`` to ``"same"``.
    """
    question = str(item["question"]).strip()

    expect_book_id = item.get("expect_book_id")
    if expect_book_id is not None:
        expect_book_id = str(expect_book_id)

    expect_page = item.get("expect_page")
    if expect_page is not None:
        try:
            expect_page = int(expect_page)
        except (TypeError, ValueError):
            logger.warning("Q%d: invalid 'expect_page' %r; ignoring", idx, expect_page)
            expect_page = None

    expect_chunk_id = item.get("expect_chunk_id")
    if expect_chunk_id is not None:
        expect_chunk_id = str(expect_chunk_id)

    book_ids = item.get("book_ids")
    if book_ids is not None and not isinstance(book_ids, list):
        logger.warning("Q%d: 'book_ids' is not a list; ignoring", idx)
        book_ids = None
    book_ids = [str(b) for b in book_ids] if book_ids else None

    lang = item.get("lang")
    if not isinstance(lang, str) or not lang.strip():
        lang = detect_lang(question)
    qtype = item.get("qtype")
    if qtype not in ("same", "cross"):
        qtype = "same"

    return {
        "question": question,
        "expect_book_id": expect_book_id,
        "expect_page": expect_page,
        "expect_chunk_id": expect_chunk_id,
        "book_ids": book_ids,
        "lang": lang,
        "qtype": qtype,
        # Lenient tier scoreable; the strict tier additionally needs a chunk id.
        "scoreable": expect_book_id is not None or expect_page is not None,
    }


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
    return rank_of_first_match(results, expect_book_id, expect_page) is not None


def rank_of_first_match(
    results: list["SearchResult"],
    expect_book_id: str | None,
    expect_page: int | None,
) -> int | None:
    """1-based rank of the first result matching the lenient gold, else None.

    Lenient tier: same book and — when a page is given — page span containing
    the gold page (:func:`_result_matches`). With no expectations at all every
    result matches, so the rank is 1 whenever anything was retrieved
    (smoke-test semantics, mirroring :func:`is_hit`).
    """
    for rank, result in enumerate(results, start=1):
        if _result_matches(result, expect_book_id, expect_page):
            return rank
    return None


def rank_of_chunk(
    results: list["SearchResult"], expect_chunk_id: str | None
) -> int | None:
    """1-based rank of the exact gold chunk id (strict tier), else None.

    Context expansion preserves each result's original ``chunk_id`` (only the
    text and page span grow), so strict matching stays valid on the expanded
    results the pipeline returns. Returns None when ``expect_chunk_id`` is
    unset.
    """
    if not expect_chunk_id:
        return None
    for rank, result in enumerate(results, start=1):
        if result.chunk_id == expect_chunk_id:
            return rank
    return None


# --- metrics ----------------------------------------------------------------


def compute_metrics(ranks: list[int | None], k: int) -> dict[str, Any]:
    """Aggregate 1-based gold ranks into recall@k / MRR@k / nDCG@k.

    All three metrics are TRUNCATED at ``k``: a gold ranked beyond ``k`` (or
    never retrieved, ``rank is None``) contributes 0 to recall, to MRR *and*
    to nDCG. Truncated MRR@k — rather than crediting ``1/rank`` past the
    cutoff — keeps every metric a function of the top-``k`` list the answer
    model actually sees. There is a single gold per question, so IDCG = 1 and
    nDCG@k is the mean of ``1 / log2(1 + rank)``.

    Returns ``{"recall", "mrr", "ndcg", "hits", "total"}``; the ratios are
    0.0 when ``ranks`` is empty.
    """
    total = len(ranks)
    kept = [r for r in ranks if r is not None and r <= k]
    if total == 0:
        return {"recall": 0.0, "mrr": 0.0, "ndcg": 0.0, "hits": 0, "total": 0}
    return {
        "recall": len(kept) / total,
        "mrr": sum(1.0 / r for r in kept) / total,
        "ndcg": sum(1.0 / math.log2(1 + r) for r in kept) / total,
        "hits": len(kept),
        "total": total,
    }


# --- settings overrides (ablations) ------------------------------------------

# Preset ablation configs: name -> settings overrides. "legacy" is special —
# it bypasses the pipeline entirely (see _retrieve_legacy).
ABLATION_PRESETS: dict[str, dict[str, Any]] = {
    "full": {},                                        # the pipeline as configured
    "no-multi-query": {"enable_multi_query": False},   # no LLM variants / RRF fusion
    "no-expansion": {"context_window_chunks": 0},      # no small-to-big stitching
    "no-floor": {"rerank_min_score": 0.0},             # keep all reranked results
    "legacy": {},  # old direct path: embed -> hybrid_search -> rerank, nothing else
}


@contextmanager
def override_settings(**overrides: Any) -> Iterator[None]:
    """Temporarily set attributes on the global ``config.settings`` object.

    The pipeline reads every knob from ``settings`` at call time, so flipping
    attributes here reconfigures retrieval without reloading models. Originals
    are restored in a ``finally`` (also on exceptions). Unknown attribute
    names raise :class:`AttributeError` before anything is mutated (pydantic
    itself would raise ``ValueError`` mid-way otherwise).
    """
    unknown = [name for name in overrides if not hasattr(settings, name)]
    if unknown:
        raise AttributeError(
            f"Unknown settings attribute(s): {', '.join(sorted(unknown))}"
        )
    originals = {name: getattr(settings, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(settings, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(settings, name, value)


def parse_set_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``--set name=value`` pairs into a settings-override dict.

    Pure function (unit-tested). Each value is coerced to the type of the
    CURRENT attribute on ``config.settings``: bool (``true/false/1/0/yes/no/
    on/off``, case-insensitive), int, or float; any other attribute type
    receives the raw string. Raises :class:`ValueError` on a malformed pair,
    an unknown attribute name, or an uncoercible value (the CLI exits 2).
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        name, sep, raw = pair.partition("=")
        name = name.strip()
        raw = raw.strip()
        if not sep or not name:
            raise ValueError(f"--set expects name=value, got {pair!r}")
        if not hasattr(settings, name):
            raise ValueError(f"--set {name}: unknown settings attribute")
        current = getattr(settings, name)
        value: Any
        if isinstance(current, bool):  # before int — bool is an int subclass
            lowered = raw.lower()
            if lowered in ("true", "1", "yes", "on"):
                value = True
            elif lowered in ("false", "0", "no", "off"):
                value = False
            else:
                raise ValueError(f"--set {name}: expected a boolean, got {raw!r}")
        elif isinstance(current, int):
            try:
                value = int(raw)
            except ValueError as exc:
                raise ValueError(f"--set {name}: expected an int, got {raw!r}") from exc
        elif isinstance(current, float):
            try:
                value = float(raw)
            except ValueError as exc:
                raise ValueError(f"--set {name}: expected a float, got {raw!r}") from exc
        else:
            value = raw
        overrides[name] = value
    return overrides


# --- expansion cache ----------------------------------------------------------


def memoize_expansion(
    func: Callable[..., list[str]], cache: dict[str, list[str]]
) -> Callable[..., list[str]]:
    """Wrap ``expand_queries`` so each unique question is expanded at most once.

    ``cache`` maps ``"v{multi_query_variants}|{question}"`` to the variant
    list (the variant count is part of the key so a ``--set
    multi_query_variants=N`` config never reuses another count's variants) and
    is mutated in place, so it round-trips through JSON and is shared across
    configs and reruns — every config retrieves with IDENTICAL query variants
    (fairness) and repeat runs cost zero expansion LLM calls. Empty results
    are NOT cached: ``expand_queries`` returns ``[]`` on any LLM failure, and
    persisting that would silently bake "no expansion" for the question into
    every later config and rerun. Copies are returned so callers cannot
    mutate the cached lists.
    """

    @functools.wraps(func)
    def wrapper(question: str, *args: Any, **kwargs: Any) -> list[str]:
        key = f"v{settings.multi_query_variants}|{question}"
        if key not in cache:
            variants = list(func(question, *args, **kwargs))
            if not variants:  # transient LLM failure — treat as a miss
                logger.warning("Expansion returned no variants for %r", question)
                return []
            cache[key] = variants
        return list(cache[key])

    return wrapper


def _load_expansion_cache(path: Path) -> dict[str, list[str]]:
    """Read a question -> variants JSON cache; missing/corrupt files yield {}."""
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Ignoring unreadable expansion cache %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("Ignoring malformed expansion cache %s (not an object)", path)
        return {}
    return {
        str(question): [str(v) for v in variants]
        for question, variants in data.items()
        if isinstance(variants, list)
    }


def _save_expansion_cache(path: Path, cache: dict[str, list[str]]) -> None:
    """Persist the expansion cache as UTF-8 JSON (best-effort, parents created)."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    except OSError as exc:
        logger.warning("Could not save expansion cache %s: %s", path, exc)


# --- model memoization (across ablation configs) -------------------------------


class _MemoEmbedder:
    """Cache query embeddings by text across ablation configs — and runs.

    Embedding is deterministic per text, and on a CPU box each ``embed_query``
    costs tens of seconds — but every config embeds the same question (and the
    same cached expansion variants). Wrapping the embedder makes the second
    and later configs free; with ``persist_path`` the cache also round-trips
    to disk so knob-grid runs on the same question set cost zero embeddings.
    Other attributes pass through untouched.
    """

    def __init__(self, embedder: Any, persist_path: Path | None = None) -> None:
        self._embedder = embedder
        self._cache: dict[str, Any] = {}
        self._path = persist_path
        self._dirty = False
        if persist_path and persist_path.is_file():
            from core.models import Embedding  # cheap pydantic model

            try:
                raw = json.loads(persist_path.read_text(encoding="utf-8"))
                self._cache = {
                    t: Embedding.model_validate(d) for t, d in raw.items()
                }
                logger.info(
                    "Loaded %d cached embedding(s) from %s",
                    len(self._cache),
                    persist_path,
                )
            except Exception as exc:  # noqa: BLE001 - cache is an optimization
                logger.warning("Ignoring unreadable embed cache %s: %s",
                               persist_path, exc)
                self._cache = {}

    def embed_query(self, text: str) -> Any:
        if text not in self._cache:
            self._cache[text] = self._embedder.embed_query(text)
            self._dirty = True
        return self._cache[text]

    def save(self) -> None:
        """Best-effort persist (no-op without a path or new entries)."""
        if not self._path or not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {t: e.model_dump() for t, e in self._cache.items()},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save embed cache %s: %s", self._path, exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._embedder, name)


class _MemoReranker:
    """Cache cross-encoder scores by ``(query, chunk_id)`` across configs.

    A score depends only on the (query, chunk text) pair, so when every
    incoming candidate already has a cached score the model is not touched:
    cached scores are attached to the FRESH result objects (never aliasing
    objects between configs — later stages mutate results in place), then
    sorted and truncated exactly like the real reranker.
    """

    _SEP = "\x1f"  # unit separator: joins (query, chunk_id) into a JSON key

    def __init__(self, reranker: Any, persist_path: Path | None = None) -> None:
        self._reranker = reranker
        self._scores: dict[tuple[str, str], float | None] = {}
        self._path = persist_path
        self._dirty = False
        if persist_path and persist_path.is_file():
            try:
                raw = json.loads(persist_path.read_text(encoding="utf-8"))
                for key, score in raw.items():
                    query, _, chunk_id = key.partition(self._SEP)
                    self._scores[(query, chunk_id)] = score
                logger.info(
                    "Loaded %d cached rerank score(s) from %s",
                    len(self._scores),
                    persist_path,
                )
            except Exception as exc:  # noqa: BLE001 - cache is an optimization
                logger.warning("Ignoring unreadable rerank cache %s: %s",
                               persist_path, exc)
                self._scores = {}

    def rerank(
        self, query: str, results: list["SearchResult"], top_k: int = 8
    ) -> list["SearchResult"]:
        if not results:
            return self._reranker.rerank(query, results, top_k=top_k)
        if any((query, r.chunk_id) not in self._scores for r in results):
            scored = self._reranker.rerank(query, list(results), top_k=len(results))
            for r in scored:
                self._scores[(query, r.chunk_id)] = r.rerank_score
            self._dirty = True
        for r in results:
            r.rerank_score = self._scores[(query, r.chunk_id)]
        ordered = sorted(results, key=lambda r: r.rerank_score or 0.0, reverse=True)
        return ordered[: max(top_k, 0)]

    def save(self) -> None:
        """Best-effort persist (no-op without a path or new entries)."""
        if not self._path or not self._dirty:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(
                json.dumps(
                    {f"{q}{self._SEP}{c}": s for (q, c), s in self._scores.items()},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.warning("Could not save rerank cache %s: %s", self._path, exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reranker, name)


# --- retrieval paths ----------------------------------------------------------


def _retrieve_legacy(
    question: str,
    *,
    embedder: "BGEM3Embedder",
    store: "QdrantStore",
    reranker: "Reranker",
    book_ids: list[str] | None,
    k: int,
) -> list["SearchResult"]:
    """The pre-pipeline direct path: embed -> hybrid search -> rerank.

    No routing, no multi-query fusion, no score floor, no context expansion —
    this script's original behaviour, kept as the ``legacy`` ablation
    baseline. ``top_k`` is passed explicitly because ``hybrid_search`` binds
    its default from ``settings`` at import time (an override would be
    invisible to the default argument).
    """
    query_emb = embedder.embed_query(question)
    candidates = store.hybrid_search(
        query_emb, top_k=settings.search_top_k, book_ids=book_ids
    )
    return reranker.rerank(question, candidates, top_k=k)


# --- evaluation -------------------------------------------------------------


def evaluate(
    questions: list[dict[str, Any]],
    *,
    k: int,
    configs: list[tuple[str, dict[str, Any]]] | None = None,
    show_answer: bool = False,
    expansion_cache_path: str | Path | None = None,
    model_cache_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run every ablation config over ``questions``; print and return a report.

    Heavy models load ONCE and are shared across configs; each config applies
    its settings overrides via :func:`override_settings` (the ``legacy``
    config bypasses the pipeline instead). Query expansion is memoized — and
    persisted to ``expansion_cache_path`` — so all configs see identical
    variants. With ``model_cache_dir``, query embeddings and rerank scores
    also persist to disk (``embed_cache.json`` / ``rerank_cache.json``), so
    repeat runs over the same question set skip the expensive CPU model calls
    entirely. ``show_answer`` prints grounded answers for the LAST config only.

    Returns ``{"k": k, "configs": {name: {"metrics", "slices",
    "per_question"}}}`` (see :func:`_summarize_config`).
    """
    # Lazy heavy imports — only when we actually run an evaluation.
    import retrieval.expand as expand_mod
    from ingest.embed import BGEM3Embedder
    from retrieval.rerank import Reranker
    from retrieval.search import QdrantStore

    configs = configs or [("full", {})]
    cache_dir = Path(model_cache_dir) if model_cache_dir else None
    embedder = _MemoEmbedder(
        BGEM3Embedder(),
        persist_path=cache_dir / "embed_cache.json" if cache_dir else None,
    )
    store = QdrantStore()
    reranker = _MemoReranker(
        Reranker(),
        persist_path=cache_dir / "rerank_cache.json" if cache_dir else None,
    )

    cache_path = Path(expansion_cache_path) if expansion_cache_path else None
    cache = _load_expansion_cache(cache_path) if cache_path else {}
    real_expand = expand_mod.expand_queries
    expand_mod.expand_queries = memoize_expansion(real_expand, cache)

    report: dict[str, Any] = {"k": k, "configs": {}}
    try:
        for pos, (name, overrides) in enumerate(configs):
            knobs = ", ".join(f"{n}={v}" for n, v in overrides.items())
            if name == "legacy":
                knobs = "direct path: embed -> hybrid_search -> rerank"
            print(f"\n### config '{name}'" + (f" ({knobs})" if knobs else ""))
            with override_settings(**overrides):
                records = _run_config(
                    questions,
                    k=k,
                    embedder=embedder,
                    store=store,
                    reranker=reranker,
                    legacy=(name == "legacy"),
                    show_answer=show_answer and pos == len(configs) - 1,
                )
            summary = _summarize_config(records, k)
            report["configs"][name] = summary
            _print_config_line(name, summary["metrics"], k)
    finally:
        expand_mod.expand_queries = real_expand
        if cache_path:
            _save_expansion_cache(cache_path, cache)
        embedder.save()
        reranker.save()

    _print_summary(report)
    return report


def _run_config(
    questions: list[dict[str, Any]],
    *,
    k: int,
    embedder: "BGEM3Embedder",
    store: "QdrantStore",
    reranker: "Reranker",
    legacy: bool,
    show_answer: bool,
) -> list[dict[str, Any]]:
    """Run one config over all questions; print and return per-question records."""
    from retrieval.pipeline import retrieve
    from retrieval.route import classify_route

    records: list[dict[str, Any]] = []
    print(f"Evaluating {len(questions)} question(s) | recall@{k}")
    print("=" * 72)

    for idx, item in enumerate(questions, start=1):
        spec = _question_spec(item, idx)
        question = spec["question"]
        book_ids = spec["book_ids"]

        if legacy:
            route_label = "legacy"
        else:
            route = classify_route(question, book_ids)
            route_label = route.value

        results: list["SearchResult"] = []
        try:
            if legacy:
                results = _retrieve_legacy(
                    question,
                    embedder=embedder,
                    store=store,
                    reranker=reranker,
                    book_ids=book_ids,
                    k=k,
                )
            else:
                results = retrieve(
                    question,
                    embedder=embedder,
                    store=store,
                    reranker=reranker,
                    route=route,
                    book_ids=book_ids,
                    rerank_top_k=k,
                )
        except Exception as exc:  # noqa: BLE001 - report and continue; counts as a miss
            logger.error("Q%d failed during retrieval: %s", idx, exc)
            print(f"[Q{idx}] ERROR: {exc}")

        rank_lenient = rank_of_first_match(
            results, spec["expect_book_id"], spec["expect_page"]
        )
        rank_strict = rank_of_chunk(results, spec["expect_chunk_id"])

        records.append(
            {
                "question": question,
                "lang": spec["lang"],
                "qtype": spec["qtype"],
                "route": route_label,
                "scoreable": spec["scoreable"],
                "strict_scoreable": spec["expect_chunk_id"] is not None,
                "rank_lenient": rank_lenient,
                "rank_strict": rank_strict,
                "n_results": len(results),
                "results": [
                    {
                        "rerank_score": r.rerank_score,
                        "is_gold": spec["scoreable"]
                        and _result_matches(
                            r, spec["expect_book_id"], spec["expect_page"]
                        ),
                        "chunk_id": r.chunk_id,
                    }
                    for r in results
                ],
            }
        )

        _print_question_result(
            idx=idx,
            spec=spec,
            route=route_label,
            rank_lenient=rank_lenient,
            rank_strict=rank_strict,
            k=k,
            results=results,
        )

        if show_answer:
            _print_answer(question, results, item.get("model"), route_label)

        print("-" * 72)

    return records


def _summarize_config(records: list[dict[str, Any]], k: int) -> dict[str, Any]:
    """Aggregate per-question records into overall metrics + lang/qtype slices.

    ``metrics`` holds the lenient-tier numbers plus a nested ``strict`` tier
    (questions carrying ``expect_chunk_id``) and ``mean_results`` over ALL
    questions; ``slices`` holds lenient metrics bucketed by ``lang`` and
    ``qtype``.
    """
    lenient = [r["rank_lenient"] for r in records if r["scoreable"]]
    strict = [r["rank_strict"] for r in records if r["strict_scoreable"]]

    metrics: dict[str, Any] = compute_metrics(lenient, k)
    metrics["strict"] = compute_metrics(strict, k)
    metrics["mean_results"] = (
        sum(r["n_results"] for r in records) / len(records) if records else 0.0
    )

    slices: dict[str, dict[str, Any]] = {}
    for dim in ("lang", "qtype"):
        buckets: dict[str, list[int | None]] = {}
        for r in records:
            if r["scoreable"]:
                buckets.setdefault(r[dim], []).append(r["rank_lenient"])
        slices[dim] = {
            label: compute_metrics(ranks, k)
            for label, ranks in sorted(buckets.items())
        }
    return {"metrics": metrics, "slices": slices, "per_question": records}


# --- pretty printing ----------------------------------------------------------


def _print_question_result(
    *,
    idx: int,
    spec: dict[str, Any],
    route: str,
    rank_lenient: int | None,
    rank_strict: int | None,
    k: int,
    results: list["SearchResult"],
) -> None:
    """Pretty-print one question's outcome and its retrieved sources.

    Per-result gold markers: ``*`` = lenient gold (book/page span), ``S`` =
    strict gold (the exact expected chunk).
    """
    if spec["scoreable"]:
        marker = f"HIT @{rank_lenient}" if rank_lenient and rank_lenient <= k else "MISS"
    elif spec["expect_chunk_id"] is None:
        marker = "----"
    else:
        marker = ""
    if spec["expect_chunk_id"]:
        strict = (
            f"strict HIT @{rank_strict}"
            if rank_strict and rank_strict <= k
            else "strict MISS"
        )
        marker = f"{marker} | {strict}" if marker else strict
    print(f"[Q{idx}] [{marker}] (route={route}) {spec['question']}")

    expect_bits: list[str] = []
    if spec["expect_book_id"] is not None:
        expect_bits.append(f"book_id={spec['expect_book_id']}")
    if spec["expect_page"] is not None:
        expect_bits.append(f"page={spec['expect_page']}")
    if spec["expect_chunk_id"] is not None:
        expect_bits.append(f"chunk_id={spec['expect_chunk_id']}")
    expect_bits.append(f"lang={spec['lang']}")
    expect_bits.append(f"qtype={spec['qtype']}")
    print("       expect: " + ", ".join(expect_bits))

    if not results:
        print("       (no results retrieved)")
        return

    for rank, r in enumerate(results, start=1):
        score = r.rerank_score if r.rerank_score is not None else r.score
        if spec["expect_chunk_id"] and r.chunk_id == spec["expect_chunk_id"]:
            flag = "S"
        elif spec["scoreable"] and _result_matches(
            r, spec["expect_book_id"], spec["expect_page"]
        ):
            flag = "*"
        else:
            flag = " "
        print(
            f"      {flag}{rank:>2}. score={score:.4f} "
            f"book={r.book_id} pages={r.page_start}-{r.page_end} "
            f'"{_snippet(r.title)}"'
        )


def _print_config_line(name: str, metrics: dict[str, Any], k: int) -> None:
    """One-line recap right after a config finishes (table comes at the end)."""
    strict = metrics["strict"]
    print("=" * 72)
    print(
        f"[{name}] recall@{k}={metrics['recall']:.3f} "
        f"({metrics['hits']}/{metrics['total']}) "
        f"MRR@{k}={metrics['mrr']:.3f} nDCG@{k}={metrics['ndcg']:.3f} "
        f"strict@{k}={strict['recall']:.3f} ({strict['hits']}/{strict['total']}) "
        f"mean-res={metrics['mean_results']:.1f}"
    )


def _ratio_cell(metrics: dict[str, Any]) -> str:
    """``0.850 (17/20)`` — a ratio with its raw counts, per the house style."""
    return f"{metrics['recall']:.3f} ({metrics['hits']}/{metrics['total']})"


def _print_summary(report: dict[str, Any]) -> None:
    """Print the cross-config summary table: overall + lang/qtype sub-rows."""
    k = report["k"]
    width = 104
    print()
    print("=" * width)
    print(f"Summary (k={k})")
    print(
        f"{'config':<26} {'slice':<12} {f'recall@{k}':>16} {'MRR':>7} "
        f"{f'nDCG@{k}':>8} {f'strict@{k}':>16} {'mean-res':>9}"
    )
    print("-" * width)
    for name, cfg in report["configs"].items():
        m = cfg["metrics"]
        print(
            f"{name:<26} {'overall':<12} {_ratio_cell(m):>16} {m['mrr']:>7.3f} "
            f"{m['ndcg']:>8.3f} {_ratio_cell(m['strict']):>16} "
            f"{m['mean_results']:>9.1f}"
        )
        for dim in ("lang", "qtype"):
            for label, sm in cfg["slices"][dim].items():
                print(
                    f"{name:<26} {f'{dim}={label}':<12} {_ratio_cell(sm):>16} "
                    f"{sm['mrr']:>7.3f} {sm['ndcg']:>8.3f} {'-':>16} {'-':>9}"
                )
    print("=" * width)


def _print_answer(
    question: str,
    results: list["SearchResult"],
    model: str | None,
    route_label: str = "local",
) -> None:
    """Generate and print a grounded Claude answer for ``question``.

    The question's classified route is forwarded so GLOBAL questions get the
    synthesis-oriented prompt ("legacy" maps to LOCAL). Best-effort: failures
    (missing key, network, optional module) are logged and reported inline so
    the recall metric is never lost.
    """
    try:
        from llm.answer import answer_question
        from retrieval.route import Route

        route = Route.GLOBAL if route_label == "global" else Route.LOCAL
        answer = answer_question(question, results, model=model, route=route)
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
        description=(
            "Evaluate retrieval quality (recall@k / MRR / nDCG, lenient + "
            "strict tiers, lang/qtype slices) over a JSONL question set, "
            "optionally across several ablation configs in one process."
        ),
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
        "--ablations",
        default=None,
        metavar="NAMES",
        help=(
            "Comma-separated configs to run in one process: "
            + ", ".join(ABLATION_PRESETS)
            + ". Default: full (or just the --set config when --set is given alone)."
        ),
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Shorthand for including the 'legacy' direct-path config.",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help=(
            "Override one settings knob (repeatable); the values form an extra "
            "config labeled by the overrides. Coerced to the attribute's type."
        ),
    )
    parser.add_argument(
        "--expansion-cache",
        default="data/eval/expansion_cache.json",
        metavar="PATH",
        help=(
            "JSON cache of LLM query expansions, shared across configs and "
            "reruns (default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--model-cache",
        default="data/eval",
        metavar="DIR",
        help=(
            "Directory for persistent embed_cache.json / rerank_cache.json "
            "(query embeddings + cross-encoder scores reused across reruns; "
            "pass an empty string to disable; default: %(default)s)."
        ),
    )
    parser.add_argument(
        "--json",
        default=None,
        metavar="PATH",
        help="Write the full report (metrics, slices, per-question) as JSON.",
    )
    parser.add_argument(
        "--dump-scores",
        default=None,
        metavar="PATH",
        help=(
            "Write per-config, per-question [rerank_score, is_gold, chunk_id] "
            "lists (powers the score-floor calibration)."
        ),
    )
    parser.add_argument(
        "--answer",
        action="store_true",
        help="Also generate and print the grounded answer per question (last config).",
    )
    return parser.parse_args(argv)


def _resolve_configs(args: argparse.Namespace) -> list[tuple[str, dict[str, Any]]]:
    """Turn --ablations/--legacy/--set into an ordered ``(name, overrides)`` list.

    Defaults to the single ``full`` config. ``--legacy`` appends the legacy
    direct-path config (given alone, it runs just that). ``--set`` pairs form
    one extra config labeled by its overrides; given alone they replace the
    ``full`` default. Raises :class:`ValueError` for unknown ablation names or
    bad ``--set`` pairs (the CLI exits 2).
    """
    names: list[str] = []
    if args.ablations:
        names = [n.strip() for n in args.ablations.split(",") if n.strip()]
    if args.legacy and "legacy" not in names:
        names.append("legacy")
    if not names and not args.set:
        names = ["full"]

    configs: list[tuple[str, dict[str, Any]]] = []
    for name in dict.fromkeys(names):  # de-dup, keep order
        if name not in ABLATION_PRESETS:
            known = ", ".join(ABLATION_PRESETS)
            raise ValueError(f"Unknown ablation config {name!r} (known: {known}).")
        configs.append((name, dict(ABLATION_PRESETS[name])))

    if args.set:
        overrides = parse_set_overrides(args.set)
        label = "set:" + ",".join(f"{n}={v}" for n, v in overrides.items())
        configs.append((label, overrides))
    return configs


def _write_json(path: Path, payload: Any) -> None:
    """Write ``payload`` as UTF-8 JSON, creating parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


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

    try:
        configs = _resolve_configs(args)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    questions = load_questions(path)
    if not questions:
        print(f"No valid questions found in {path}.", file=sys.stderr)
        return 1

    report = evaluate(
        questions,
        k=args.k,
        configs=configs,
        show_answer=args.answer,
        expansion_cache_path=args.expansion_cache,
        model_cache_dir=args.model_cache or None,
    )

    if args.json:
        _write_json(Path(args.json), report)
        print(f"Report written to {args.json}")
    if args.dump_scores:
        scores = {
            name: [
                {"question": rec["question"], "scores": rec["results"]}
                for rec in cfg["per_question"]
            ]
            for name, cfg in report["configs"].items()
        }
        _write_json(Path(args.dump_scores), scores)
        print(f"Score dump written to {args.dump_scores}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
