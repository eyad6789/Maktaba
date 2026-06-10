import { Fragment, type ReactNode } from 'react'
import type { SearchResult } from '../api/types'
import { dirFor } from '../i18n'
import { useApp } from '../state/AppContext'
import type { UiMessage } from './MessageList'

const PROVIDER_LABELS: Record<string, string> = {
  gemini: 'Gemini',
  claude: 'Claude',
  local: 'Local',
  minimax: 'MiniMax',
  groq: 'Groq',
}

export function providerLabel(id: string | null | undefined): string {
  if (!id) return ''
  return PROVIDER_LABELS[id] ?? id
}

/** Inline renderer: `code`, **bold**, *italic*, and [n] citation superscripts. */
function renderInline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = []
  const re = /(`[^`\n]+`)|(\*\*[^*\n]+\*\*)|(\*[^*\n]+\*)|(\[\d{1,3}\])/g
  let last = 0
  let match: RegExpExecArray | null
  let i = 0
  while ((match = re.exec(text)) !== null) {
    if (match.index > last) out.push(text.slice(last, match.index))
    const tok = match[0]
    const k = `${keyBase}-${i++}`
    if (tok.startsWith('`')) out.push(<code key={k}>{tok.slice(1, -1)}</code>)
    else if (tok.startsWith('**')) out.push(<strong key={k}>{tok.slice(2, -2)}</strong>)
    else if (tok.startsWith('*')) out.push(<em key={k}>{tok.slice(1, -1)}</em>)
    else
      out.push(
        <sup className="cite" key={k} title={tok}>
          {tok}
        </sup>,
      )
    last = match.index + tok.length
  }
  if (last < text.length) out.push(text.slice(last))
  return out
}

function renderRich(content: string): ReactNode {
  const paragraphs = content.split(/\n{2,}/)
  return paragraphs.map((para, pi) => (
    <p key={pi}>
      {para.split('\n').map((line, li, arr) => (
        <Fragment key={li}>
          {renderInline(line, `${pi}-${li}`)}
          {li < arr.length - 1 && <br />}
        </Fragment>
      ))}
    </p>
  ))
}

/** Dedupe sources for display by (book, page range). */
function dedupeSources(sources: SearchResult[]): SearchResult[] {
  const seen = new Set<string>()
  const out: SearchResult[] = []
  for (const s of sources) {
    const k = `${s.book_id}:${s.page_start}:${s.page_end}`
    if (!seen.has(k)) {
      seen.add(k)
      out.push(s)
    }
  }
  return out
}

interface Props {
  msg: UiMessage
  dir: 'rtl' | 'ltr'
  onUseAuto: () => void
  onRetry: () => void
}

export default function MessageBubble({ msg, dir, onUseAuto, onRetry }: Props) {
  const { d, lang } = useApp()

  if (msg.role === 'user') {
    return (
      <div className="msg msg--user" dir={dir}>
        {msg.content}
      </div>
    )
  }

  const sources = msg.sources ? dedupeSources(msg.sources) : []

  return (
    <div className={`msg msg--assistant ${msg.streaming ? 'streaming' : ''}`} dir={dir}>
      {msg.searching && (
        <div className="msg__status">
          <span className="quill" /> {d.searchingBooks}
        </div>
      )}

      {!msg.searching && msg.streaming && !msg.content && !msg.error && (
        <div className="msg__status">
          <span className="quill" /> {d.thinking}
        </div>
      )}

      {msg.content && (
        <div className="msg__body">
          {renderRich(msg.content)}
          {msg.streaming && <span className="caret" />}
        </div>
      )}

      {msg.error && (
        <div className="msg--error" dir={dirFor(msg.error.message, lang)}>
          <span>⚠ {msg.error.message}</span>
          <span className="actions">
            {msg.error.provider && (
              <button type="button" className="btn-small" onClick={onUseAuto}>
                {d.useAuto}
              </button>
            )}
            <button type="button" className="btn-small" onClick={onRetry}>
              {d.retry}
            </button>
          </span>
        </div>
      )}

      {!msg.streaming && !msg.error && (msg.provider || msg.grounded === false) && (
        <div className="msg__meta">
          {msg.provider && (
            <span className="tag">
              <span className="dot dot--ok" />
              {d.answeredBy} {providerLabel(msg.provider)}
              {msg.model ? ` · ${msg.model}` : ''}
            </span>
          )}
          {msg.grounded === false && <span className="tag tag--warn">{d.ungrounded}</span>}
        </div>
      )}

      {!msg.streaming && sources.length > 0 && (
        <details className="sources">
          <summary>
            {d.sources} ({sources.length})
          </summary>
          <ol>
            {sources.map((s, i) => (
              <li key={s.chunk_id} dir={dirFor(s.title, lang)}>
                <span className="src-n">[{i + 1}]</span> {s.title}
                {s.author ? ` — ${s.author}` : ''}{' '}
                <span className="src-pages">
                  · {s.page_start === s.page_end ? `p. ${s.page_start}` : `pp. ${s.page_start}–${s.page_end}`}
                </span>
              </li>
            ))}
          </ol>
        </details>
      )}
    </div>
  )
}
