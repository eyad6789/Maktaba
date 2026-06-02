"""Cross-encoder reranking of hybrid-search candidates.

`Reranker` wraps `FlagEmbedding.FlagReranker` (bge-reranker-v2-m3) to rescore
the (query, chunk) pairs produced by `QdrantStore.hybrid_search`. The first
retrieval pass favours recall; this second pass restores precision by scoring
each candidate against the query with a cross-encoder, then keeping the best
``top_k``.

`FlagEmbedding` (and its torch/transformers dependencies) is imported lazily so
this module stays importable — and `py_compile`-clean — on a CPU-only box that
has not installed the model stack.
"""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

from config import settings
from core.logging import get_logger
from core.models import SearchResult

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from FlagEmbedding import FlagReranker

logger = get_logger(__name__)


class Reranker:
    """Cross-encoder reranker over `SearchResult` candidates.

    The heavy `FlagReranker` model is loaded lazily on first use so constructing
    a `Reranker` is cheap and side-effect free.
    """

    def __init__(
        self,
        model_name: str = settings.reranker_model,
        device: str = settings.reranker_device,
        use_fp16: bool = settings.reranker_use_fp16,
    ) -> None:
        self.model_name = model_name
        self.device = device
        # fp16 is only meaningful on CUDA; force it off on CPU to avoid errors.
        self.use_fp16 = bool(use_fp16) and device != "cpu"
        self._reranker: "FlagReranker | None" = None

    # -- model ----------------------------------------------------------------

    @property
    def reranker(self) -> "FlagReranker":
        """Lazily construct and cache the underlying FlagReranker."""
        if self._reranker is None:
            from FlagEmbedding import FlagReranker

            logger.info(
                "Loading reranker %s (device=%s, fp16=%s)",
                self.model_name,
                self.device,
                self.use_fp16,
            )
            self._reranker = FlagReranker(
                self.model_name,
                use_fp16=self.use_fp16,
                devices=self.device,
            )
        return self._reranker

    # -- public API -----------------------------------------------------------

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int = settings.rerank_top_k,
    ) -> list[SearchResult]:
        """Rescore ``results`` against ``query`` and return the top ``top_k``.

        Builds ``(query, result.text)`` pairs, scores them with the
        cross-encoder (normalized to 0..1), writes ``rerank_score`` onto each
        result, sorts descending, and truncates to ``top_k``. Returns an empty
        list when ``results`` is empty.
        """
        if not results:
            return []

        pairs = [[query, r.text] for r in results]
        scores = self.reranker.compute_score(pairs, normalize=True)

        # compute_score may return a single scalar for one pair (including a
        # numpy scalar, which is a numbers.Real but not a Python int/float and
        # is *not* iterable), else a sequence/ndarray of scores.
        if isinstance(scores, numbers.Real):
            scores = [float(scores)]
        else:
            scores = [float(s) for s in scores]

        for result, score in zip(results, scores):
            result.rerank_score = score

        ranked = sorted(
            results,
            key=lambda r: (r.rerank_score if r.rerank_score is not None else float("-inf")),
            reverse=True,
        )

        limit = top_k if top_k is not None and top_k > 0 else len(ranked)
        top = ranked[:limit]
        logger.debug(
            "Reranked %d candidates -> kept %d (top score=%.4f)",
            len(results),
            len(top),
            top[0].rerank_score if top and top[0].rerank_score is not None else 0.0,
        )
        return top
