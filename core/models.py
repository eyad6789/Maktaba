"""Shared data models — the contract every module reads and writes.

These pydantic models are the canonical types passed between the ingestion
pipeline, the vector store, the retriever, and the answer layer. Do not
duplicate these shapes elsewhere; import them from here.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class PageKind(str, Enum):
    """How a PDF page's text was (or will be) obtained."""

    NATIVE = "native"      # extractable digital text (PyMuPDF)
    SCANNED = "scanned"    # image page requiring OCR
    EMPTY = "empty"        # no text and no meaningful image


class BookMeta(BaseModel):
    """Identity + provenance for one ingested book."""

    book_id: str                       # stable id, e.g. uuid5 of file_hash
    title: str
    author: str | None = None
    language: str | None = None        # "ar" | "en" | "mixed"
    source_path: str
    num_pages: int = 0
    file_hash: str                     # sha256 of the file bytes (dedup key)


class PageContent(BaseModel):
    """Text for a single page after extraction or OCR."""

    page_number: int                   # 1-based
    text: str
    kind: PageKind
    lang: str | None = None            # detected language of the page


class Chunk(BaseModel):
    """A retrievable unit of text with citation metadata.

    `text` holds the NORMALIZED text used for embedding and shown to the LLM.
    `page_start`/`page_end` enable [title, page] citations.
    """

    chunk_id: str                      # stable: uuid5(f"{book_id}:{chunk_index}")
    book_id: str
    title: str
    author: str | None = None
    text: str
    page_start: int
    page_end: int
    chunk_index: int
    lang: str | None = None
    token_count: int = 0
    # Hierarchical comprehension layer. "passage" = a raw retrievable chunk
    # (the default, so every existing chunk stays valid); "chapter_summary" and
    # "book_summary" are nodes produced at ingest time and stored alongside the
    # raw chunks. `parent_id` links a node to its parent (chapter -> book);
    # `chapter_title` labels a summary node's section.
    level: str = "passage"
    parent_id: str | None = None
    chapter_title: str | None = None


class SparseVector(BaseModel):
    """Sparse (lexical) vector in index/value form, as Qdrant expects."""

    indices: list[int]
    values: list[float]


class Embedding(BaseModel):
    """BGE-M3 output: dense + sparse for hybrid search."""

    dense: list[float]
    sparse: SparseVector


class SearchResult(BaseModel):
    """One retrieved chunk with scores. `rerank_score` set after reranking."""

    chunk_id: str
    score: float                       # fused retrieval score (RRF)
    text: str
    book_id: str
    title: str
    author: str | None = None
    page_start: int
    page_end: int
    chunk_index: int
    lang: str | None = None
    rerank_score: float | None = None
    level: str = "passage"             # "passage" | "chapter_summary" | "book_summary"
    parent_id: str | None = None
    chapter_title: str | None = None


class Citation(BaseModel):
    title: str
    author: str | None = None
    page_start: int
    page_end: int
    book_id: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[SearchResult]
    model: str
    grounded: bool = True              # False when answer is "not found in books"
    provider: str | None = None        # provider id that answered ("gemini"|"claude"|"local"|...)


class IngestStats(BaseModel):
    """Result of ingesting one book; logged and returned by the pipeline."""

    book_id: str
    title: str
    num_pages: int = 0
    native_pages: int = 0
    scanned_pages: int = 0
    failed_pages: list[int] = Field(default_factory=list)
    num_chunks: int = 0
    num_summary_nodes: int = 0         # chapter/book comprehension nodes built
    status: str = "completed"          # "completed" | "failed" | "skipped"
    error: str | None = None


# --- API request/response models ---------------------------------------------


class QueryRequest(BaseModel):
    question: str
    top_k: int | None = None           # override reranker top_k
    book_ids: list[str] | None = None  # optional filter to specific books
    model: str | None = None           # override answer model
    route: str | None = None           # force "local"|"global" (else auto-classified)
    provider: str | None = None        # pin a provider ("gemini"|"claude"|"local") or "auto"


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[SearchResult]
    model: str
    grounded: bool
    provider: str | None = None        # provider id that actually answered


class IngestRequest(BaseModel):
    path: str                          # file or directory path on the server
    recursive: bool = True


class IngestResponse(BaseModel):
    enqueued: list[str]                # job ids or file paths accepted
    skipped: list[str]                 # already-ingested files
    count: int


# --- Chat (multi-turn) models ------------------------------------------------


class ChatMessage(BaseModel):
    role: str                          # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]        # full conversation; last must be a user turn
    top_k: int | None = None           # override reranker top_k
    book_ids: list[str] | None = None  # optional filter to specific books
    model: str | None = None           # override answer model
    condense: bool = True              # rewrite follow-ups into a standalone query
    route: str | None = None           # force "local"|"global" (else auto-classified)
    provider: str | None = None        # pin a provider ("gemini"|"claude"|"local") or "auto"


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    sources: list[SearchResult]
    model: str
    grounded: bool
    search_query: str                  # the (possibly condensed) query used to retrieve
    provider: str | None = None        # provider id that actually answered


class ChatStreamRequest(BaseModel):
    """One streamed chat turn. The server owns the conversation history."""

    conversation_id: str | None = None  # null -> server creates a new conversation
    message: str                        # the user's new message (required, non-empty)
    book_ids: list[str] | None = None   # per-turn override of the stored scope
    provider: str | None = None         # per-turn override: "auto"|"gemini"|"claude"|"local"
    top_k: int | None = None
    route: str | None = None
    condense: bool = True


# --- Conversations (persisted chat history) -----------------------------------


class ConversationCreate(BaseModel):
    title: str = ""
    model: str = "auto"                # default provider for this conversation
    book_ids: list[str] | None = None  # default book scope (None = all books)


class ConversationRename(BaseModel):
    title: str


class ConversationMessage(BaseModel):
    """One persisted chat turn as returned by GET /conversations/{id}."""

    id: int
    role: str                          # "user" | "assistant"
    content: str
    citations: list[Citation] | None = None
    model: str | None = None           # provider id that answered (assistant turns)
    grounded: bool | None = None
    created_at: str


class ConversationSummary(BaseModel):
    id: str
    title: str
    model: str
    book_ids: list[str] | None = None
    created_at: str
    updated_at: str
    message_count: int = 0


class ConversationDetail(ConversationSummary):
    messages: list[ConversationMessage] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    conversations: list[ConversationSummary]
