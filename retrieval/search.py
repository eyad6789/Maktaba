"""Qdrant-backed hybrid (dense + sparse) vector store and retriever.

`QdrantStore` owns the collection lifecycle (create + payload indexes),
upserts BGE-M3 dense/sparse vectors with citation payload, and runs hybrid
search fused with Reciprocal Rank Fusion via Qdrant's Query API.

`qdrant_client` is imported lazily so this module stays importable (and
`py_compile`-clean) on a box without the client installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from config import settings
from core import schema
from core.logging import get_logger
from core.models import Chunk, Embedding, SearchResult

if TYPE_CHECKING:  # pragma: no cover - typing only, never imported at runtime
    from qdrant_client import QdrantClient

logger = get_logger(__name__)


class QdrantStore:
    """Thin wrapper over a Qdrant collection for hybrid book retrieval."""

    def __init__(
        self,
        url: str = settings.qdrant_url,
        api_key: str | None = settings.qdrant_api_key,
        collection: str = schema.COLLECTION,
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.collection = collection
        self._client: "QdrantClient | None" = None

    # -- client ---------------------------------------------------------------

    @property
    def client(self) -> "QdrantClient":
        """Lazily construct and cache the Qdrant client."""
        if self._client is None:
            from qdrant_client import QdrantClient

            logger.debug("Connecting to Qdrant at %s", self.url)
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    # -- collection lifecycle -------------------------------------------------

    def ensure_collection(self) -> None:
        """Create the collection + payload indexes if it does not yet exist.

        Defines a named dense vector (cosine) and a named sparse vector, enables
        scalar quantization when configured, and indexes ``book_id`` and ``lang``
        for fast filtering. Idempotent: a no-op when the collection exists.
        """
        from qdrant_client import models as qm

        client = self.client
        if client.collection_exists(self.collection):
            logger.debug("Collection %r already exists; skipping create", self.collection)
            self._ensure_payload_indexes()
            return

        quantization_config = None
        if settings.qdrant_use_quantization:
            quantization_config = qm.ScalarQuantization(
                scalar=qm.ScalarQuantizationConfig(
                    type=qm.ScalarType.INT8,
                    always_ram=True,
                )
            )

        logger.info(
            "Creating Qdrant collection %r (dim=%d, quantization=%s)",
            self.collection,
            schema.DENSE_DIM,
            bool(quantization_config),
        )
        client.create_collection(
            collection_name=self.collection,
            vectors_config={
                schema.DENSE_VECTOR: qm.VectorParams(
                    size=schema.DENSE_DIM,
                    distance=qm.Distance.COSINE,
                )
            },
            sparse_vectors_config={
                schema.SPARSE_VECTOR: qm.SparseVectorParams()
            },
            quantization_config=quantization_config,
        )
        self._ensure_payload_indexes()

    def _ensure_payload_indexes(self) -> None:
        """Create keyword payload indexes for the configured filter keys."""
        from qdrant_client import models as qm

        client = self.client
        for key in schema.INDEXED_PAYLOAD_KEYS:
            try:
                client.create_payload_index(
                    collection_name=self.collection,
                    field_name=key,
                    field_schema=qm.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # index may already exist — tolerate it
                logger.debug("Payload index for %r not created: %s", key, exc)

    # -- writes ---------------------------------------------------------------

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        embeddings: list[Embedding],
        batch_size: int = 128,
    ) -> int:
        """Upsert chunks with their dense+sparse vectors and citation payload.

        Point ids are the chunk uuid strings. Payload mirrors
        ``schema.PAYLOAD_FIELDS``. Returns the number of points upserted.
        """
        if len(chunks) != len(embeddings):
            raise ValueError(
                f"chunks/embeddings length mismatch: {len(chunks)} != {len(embeddings)}"
            )
        if not chunks:
            return 0

        from qdrant_client import models as qm

        client = self.client
        total = 0
        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_embs = embeddings[start : start + batch_size]
            points: list[Any] = []
            for chunk, emb in zip(batch_chunks, batch_embs):
                payload = {field: getattr(chunk, field) for field in schema.PAYLOAD_FIELDS}
                sparse = qm.SparseVector(
                    indices=list(emb.sparse.indices),
                    values=list(emb.sparse.values),
                )
                points.append(
                    qm.PointStruct(
                        id=chunk.chunk_id,
                        vector={
                            schema.DENSE_VECTOR: list(emb.dense),
                            schema.SPARSE_VECTOR: sparse,
                        },
                        payload=payload,
                    )
                )
            client.upsert(collection_name=self.collection, points=points)
            total += len(points)
            logger.debug(
                "Upserted %d/%d chunks into %r", total, len(chunks), self.collection
            )

        logger.info("Upserted %d chunks into %r", total, self.collection)
        return total

    # -- reads ----------------------------------------------------------------

    @staticmethod
    def _build_filter(
        book_ids: list[str] | None,
        levels: list[str] | None = None,
    ) -> "Any | None":
        """Compose a Qdrant payload filter from optional book + level constraints.

        Both constraints are ``MatchAny`` (OR within a field) AND-ed together.
        Returns ``None`` when neither is given — i.e. an unfiltered search,
        identical to the original behaviour. ``qm`` is imported lazily so the
        module stays importable without ``qdrant_client``.
        """
        from qdrant_client import models as qm

        must = []
        if book_ids:
            must.append(
                qm.FieldCondition(key="book_id", match=qm.MatchAny(any=list(book_ids)))
            )
        if levels:
            must.append(
                qm.FieldCondition(key="level", match=qm.MatchAny(any=list(levels)))
            )
        return qm.Filter(must=must) if must else None

    def hybrid_search(
        self,
        query: Embedding,
        top_k: int = settings.search_top_k,
        book_ids: list[str] | None = None,
        levels: list[str] | None = None,
    ) -> list[SearchResult]:
        """Hybrid dense+sparse retrieval fused with RRF via Qdrant's Query API.

        Prefetches the top ``top_k`` from both the dense and sparse vectors, then
        fuses with Reciprocal Rank Fusion. When ``book_ids`` is given, restricts
        results to those books; when ``levels`` is given (e.g.
        ``["chapter_summary", "book_summary"]``), restricts to those node levels.
        ``levels=None`` searches every level — identical to the original
        behaviour. Returns up to ``top_k`` :class:`SearchResult`.
        """
        from qdrant_client import models as qm

        query_filter = self._build_filter(book_ids, levels)

        dense_vector = list(query.dense)
        sparse_vector = qm.SparseVector(
            indices=list(query.sparse.indices),
            values=list(query.sparse.values),
        )

        prefetch = [
            qm.Prefetch(
                query=dense_vector,
                using=schema.DENSE_VECTOR,
                limit=top_k,
                filter=query_filter,
            ),
            qm.Prefetch(
                query=sparse_vector,
                using=schema.SPARSE_VECTOR,
                limit=top_k,
                filter=query_filter,
            ),
        ]

        response = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=qm.FusionQuery(fusion=qm.Fusion.RRF),
            limit=top_k,
            with_payload=True,
            query_filter=query_filter,
        )

        results: list[SearchResult] = []
        for point in response.points:
            payload = point.payload or {}
            results.append(self._point_to_result(str(point.id), float(point.score), payload))

        logger.debug(
            "hybrid_search returned %d results (top_k=%d, book_ids=%s, levels=%s)",
            len(results),
            top_k,
            book_ids,
            levels,
        )
        return results

    def fetch_children(
        self,
        parent_ids: list[str],
        limit_per_parent: int = 6,
    ) -> list[SearchResult]:
        """Fetch the child nodes of the given parent node ids.

        Used by GLOBAL retrieval to drill from a kept chapter summary down into
        its source passages, so the answer's citations land on real text. Scans
        by the indexed ``parent_id`` payload key (no vector search). Returned
        results carry ``score=0.0`` — the caller reranks them against the query.
        """
        if not parent_ids:
            return []

        from qdrant_client import models as qm

        scroll_filter = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="parent_id", match=qm.MatchAny(any=list(parent_ids))
                )
            ]
        )
        limit = max(1, limit_per_parent * len(parent_ids))
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        results = [
            self._point_to_result(str(p.id), 0.0, p.payload or {}) for p in points
        ]
        logger.debug(
            "fetch_children returned %d children for %d parent(s)",
            len(results),
            len(parent_ids),
        )
        return results

    @staticmethod
    def _point_to_result(
        chunk_id: str, score: float, payload: dict[str, Any]
    ) -> SearchResult:
        """Map a Qdrant point payload onto a :class:`SearchResult`."""
        return SearchResult(
            chunk_id=chunk_id,
            score=score,
            text=payload.get("text", ""),
            book_id=payload.get("book_id", ""),
            title=payload.get("title", ""),
            author=payload.get("author"),
            page_start=int(payload.get("page_start", 0)),
            page_end=int(payload.get("page_end", 0)),
            chunk_index=int(payload.get("chunk_index", 0)),
            lang=payload.get("lang"),
            level=payload.get("level", "passage"),       # default: pre-Phase-2 points
            parent_id=payload.get("parent_id"),
            chapter_title=payload.get("chapter_title"),
        )

    def count(self) -> int:
        """Return the number of points (chunks) stored in the collection."""
        try:
            result = self.client.count(collection_name=self.collection, exact=True)
        except Exception as exc:
            logger.warning("count() failed for %r: %s", self.collection, exc)
            return 0
        return int(result.count)
