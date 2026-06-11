"""Qdrant collection schema constants — single source of truth for the
vector store layout. Both the store (writer) and any admin scripts import these.
"""

from __future__ import annotations

from config import settings

COLLECTION: str = settings.qdrant_collection

# Named vectors stored per point (BGE-M3 produces both).
DENSE_VECTOR: str = "dense"
SPARSE_VECTOR: str = "sparse"

DENSE_DIM: int = settings.embedding_dim        # 1024 for BGE-M3
DENSE_DISTANCE: str = "Cosine"                 # BGE-M3 dense vectors; cosine similarity

# Payload field names stored alongside each vector (mirror core.models.Chunk).
PAYLOAD_FIELDS = (
    "book_id",
    "title",
    "author",
    "text",
    "page_start",
    "page_end",
    "chunk_index",
    "lang",
    "token_count",
    "level",            # "passage" | "chapter_summary" | "book_summary"
    "parent_id",        # links a summary node to its parent (chapter -> book)
    "chapter_title",    # section label for summary nodes
)

# Payload keys we create indexes on for fast filtering. `level` enables
# level-routed retrieval; `parent_id` enables fetching a summary's children.
INDEXED_PAYLOAD_KEYS = ("book_id", "lang", "level", "parent_id")

# Integer payload indexes (separate list: different field schema). `chunk_index`
# enables fetching a passage's neighbours for small-to-big context expansion.
INDEXED_INTEGER_KEYS = ("chunk_index",)
