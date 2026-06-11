"""BGE-M3 embedder producing dense + sparse vectors for hybrid retrieval.

Wraps ``FlagEmbedding.BGEM3FlagModel`` to encode documents and queries into the
canonical :class:`core.models.Embedding` (a dense float vector plus a sparse
lexical :class:`core.models.SparseVector`). The heavy model is imported and
loaded lazily so this module stays importable on a CPU-only box.
"""

from __future__ import annotations

from typing import Any

from config import settings
from core.logging import get_logger
from core.models import Embedding, SparseVector

logger = get_logger(__name__)


class BGEM3Embedder:
    """Encode text into BGE-M3 dense + sparse vectors.

    The underlying ``BGEM3FlagModel`` is loaded lazily on first encode (or via
    :meth:`load`) to avoid importing torch/FlagEmbedding at module import time.
    """

    def __init__(
        self,
        model_name: str = settings.embedding_model,
        device: str = settings.embedding_device,
        use_fp16: bool = settings.embedding_use_fp16,
        *,
        batch_size: int = settings.embedding_batch_size,
        max_length: int = settings.embedding_max_length,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.batch_size = batch_size
        self.max_length = max_length
        self._model: Any | None = None

    # -- model lifecycle -----------------------------------------------------

    def load(self) -> Any:
        """Load and cache the FlagEmbedding model (idempotent)."""
        if self._model is None:
            from FlagEmbedding import BGEM3FlagModel  # heavy: lazy import

            logger.info(
                "Loading BGE-M3 embedder %s (device=%s, fp16=%s)",
                self.model_name,
                self.device,
                self.use_fp16,
            )
            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.use_fp16,
                device=self.device,
            )
        return self._model

    @property
    def model(self) -> Any:
        """The loaded model, loading it on first access."""
        return self.load()

    # -- encoding ------------------------------------------------------------

    def embed_documents(self, texts: list[str]) -> list[Embedding]:
        """Embed a batch of document texts into dense + sparse vectors.

        Empty / whitespace-only texts are encoded as well (BGE-M3 tolerates
        them, yielding a near-empty sparse vector); the index/length alignment
        with ``texts`` is always preserved.
        """
        if not texts:
            return []

        model = self.load()
        output = model.encode(
            texts,
            batch_size=self.batch_size,
            max_length=self.max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )

        dense_vecs = output["dense_vecs"]
        lexical_weights = output["lexical_weights"]

        embeddings: list[Embedding] = []
        for dense, weights in zip(dense_vecs, lexical_weights):
            embeddings.append(
                Embedding(
                    dense=_to_float_list(dense),
                    sparse=_lexical_to_sparse(weights),
                )
            )
        return embeddings

    def embed_queries(self, texts: list[str]) -> list[Embedding]:
        """Embed several query strings in ONE batched model call.

        On CPU a single encode of N short texts costs far less than N serial
        calls (the model amortizes tokenization + forward overhead), so the
        retrieval pipeline embeds the question and all its expansion variants
        together. Index alignment with ``texts`` is preserved.
        """
        return self.embed_documents(texts)

    def embed_query(self, text: str) -> Embedding:
        """Embed a single query string into a dense + sparse vector."""
        results = self.embed_queries([text])
        if not results:
            # Defensive: only happens for an explicitly empty input list, which
            # cannot occur here since we always pass exactly one element.
            return Embedding(dense=[], sparse=SparseVector(indices=[], values=[]))
        return results[0]


# -- helpers -----------------------------------------------------------------


def _to_float_list(vec: Any) -> list[float]:
    """Convert a numpy array / sequence of floats into a plain list[float]."""
    tolist = getattr(vec, "tolist", None)
    if callable(tolist):
        vec = tolist()
    return [float(x) for x in vec]


def _lexical_to_sparse(weights: Any) -> SparseVector:
    """Convert BGE-M3 ``lexical_weights`` into a :class:`SparseVector`.

    ``lexical_weights`` is a dict mapping a token id (str) to its weight
    (float). Token ids are parsed to ints; zero-weight entries are dropped so
    Qdrant stores a compact sparse vector.
    """
    indices: list[int] = []
    values: list[float] = []
    items = weights.items() if hasattr(weights, "items") else weights
    for token_id, weight in items:
        value = float(weight)
        if value == 0.0:
            continue
        indices.append(int(token_id))
        values.append(value)
    return SparseVector(indices=indices, values=values)
