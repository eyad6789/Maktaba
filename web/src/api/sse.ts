// SSE consumption for POST /chat/stream. EventSource can't POST or set the
// X-API-Key header, so we parse the stream off fetch()'s ReadableStream.

import { ApiError, authedFetch } from './client'
import type {
  ChatStreamRequest,
  DoneEvent,
  MetaEvent,
  ProviderEvent,
  StreamErrorEvent,
} from './types'

export interface StreamCallbacks {
  onMeta?: (meta: MetaEvent) => void
  onProvider?: (p: ProviderEvent) => void
  onDelta?: (text: string) => void
  onDone?: (done: DoneEvent) => void
  onError?: (err: StreamErrorEvent) => void
}

function dispatchFrame(frame: string, cb: StreamCallbacks): void {
  let event = 'message'
  const dataLines: string[] = []
  for (const line of frame.split('\n')) {
    if (line.startsWith('event:')) event = line.slice(6).trim()
    else if (line.startsWith('data:')) dataLines.push(line.slice(5).trimStart())
    // lines starting with ":" are SSE comments/heartbeats — ignored
  }
  if (!dataLines.length) return
  let data: unknown
  try {
    data = JSON.parse(dataLines.join('\n'))
  } catch {
    return // tolerate malformed frames rather than killing the stream
  }
  switch (event) {
    case 'meta':
      cb.onMeta?.(data as MetaEvent)
      break
    case 'provider':
      cb.onProvider?.(data as ProviderEvent)
      break
    case 'delta':
      cb.onDelta?.((data as { text: string }).text ?? '')
      break
    case 'done':
      cb.onDone?.(data as DoneEvent)
      break
    case 'error':
      cb.onError?.(data as StreamErrorEvent)
      break
  }
}

/**
 * Run one streamed chat turn. Resolves when the stream closes; SSE-level
 * errors arrive via cb.onError, HTTP-level failures throw ApiError.
 */
export async function streamChat(
  body: ChatStreamRequest,
  cb: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const res = await authedFetch('/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal,
  })
  if (!res.ok || !res.body) {
    throw new ApiError(res.status, await res.text())
  }

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    let sep: number
    while ((sep = buf.indexOf('\n\n')) !== -1) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      if (frame.trim()) dispatchFrame(frame, cb)
    }
  }
  if (buf.trim()) dispatchFrame(buf, cb) // tolerate a missing trailing blank line
}
