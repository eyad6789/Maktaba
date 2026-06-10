// TypeScript mirrors of the FastAPI shapes (core/models.py + routers).

export interface Citation {
  title: string
  author: string | null
  page_start: number
  page_end: number
  book_id: string
}

export interface SearchResult {
  chunk_id: string
  score: number
  text: string
  book_id: string
  title: string
  author: string | null
  page_start: number
  page_end: number
  chunk_index: number
  lang: string | null
  rerank_score: number | null
  level: string
  parent_id: string | null
  chapter_title: string | null
}

export interface ProviderInfo {
  id: string
  label: string
  model: string | null
  available: boolean
  chain?: string[]
}

export interface ModelsResponse {
  default: string
  providers: ProviderInfo[]
}

export interface ConversationSummary {
  id: string
  title: string
  model: string
  book_ids: string[] | null
  created_at: string
  updated_at: string
  message_count: number
}

export interface ConversationMessage {
  id: number
  role: 'user' | 'assistant'
  content: string
  citations: Citation[] | null
  model: string | null
  grounded: boolean | null
  created_at: string
}

export interface ConversationDetail extends ConversationSummary {
  messages: ConversationMessage[]
}

export interface ChatStreamRequest {
  conversation_id: string | null
  message: string
  book_ids: string[] | null
  provider: string | null
  top_k?: number | null
  route?: string | null
  condense?: boolean
}

// SSE event payloads from POST /chat/stream
export interface MetaEvent {
  conversation_id: string
  search_query: string
  route: string
  provider_requested: string
  sources: SearchResult[]
}

export interface ProviderEvent {
  provider: string
  model: string
}

export interface DoneEvent {
  citations: Citation[]
  grounded: boolean
  provider: string | null
  model: string | null
  conversation_id: string
}

export interface StreamErrorEvent {
  provider: string | null
  reason: string
  message: string
  partial: boolean
}

// Dashboard shapes (api/routes_books.py)
export interface BookRow {
  book_id: string
  title: string
  author: string | null
  language: string | null
  source_path: string
  num_pages: number
  num_chunks: number
  status: string
  error: string | null
  updated_at: string
}

export interface BooksResponse {
  books: BookRow[]
  total_books: number
  total_chunks: number
}

export interface JobInfo {
  job_id: string
  state: 'queued' | 'started' | 'finished' | 'failed' | 'not_found' | string
  stage: string | null
  current: number | null
  total: number | null
  book_id: string | null
  title: string | null
  path?: string | null
  error: string | null
  result: Record<string, unknown> | null
}

export interface UploadResponse {
  status: 'queued' | 'duplicate'
  job_id: string | null
  book_id: string
  filename: string
  size_bytes?: number
}
