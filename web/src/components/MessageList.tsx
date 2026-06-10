import { useEffect, useRef } from 'react'
import type { Citation, SearchResult, StreamErrorEvent } from '../api/types'
import { dirFor } from '../i18n'
import { useApp } from '../state/AppContext'
import MessageBubble from './MessageBubble'

/** UI-side message shape (richer than the persisted one: streaming/error states). */
export interface UiMessage {
  key: string
  role: 'user' | 'assistant'
  content: string
  citations?: Citation[]
  sources?: SearchResult[]
  provider?: string | null
  model?: string | null
  grounded?: boolean | null
  streaming?: boolean
  searching?: boolean
  error?: StreamErrorEvent | null
}

interface Props {
  messages: UiMessage[]
  onUseAuto: () => void
  onRetry: () => void
}

export default function MessageList({ messages, onUseAuto, onRetry }: Props) {
  const { lang } = useApp()
  const endRef = useRef<HTMLDivElement>(null)
  const stickToBottom = useRef(true)
  const scrollerRef = useRef<HTMLDivElement>(null)

  // Autoscroll only while the user is already near the bottom.
  useEffect(() => {
    if (stickToBottom.current) endRef.current?.scrollIntoView({ block: 'end' })
  }, [messages])

  const handleScroll = () => {
    const el = scrollerRef.current
    if (!el) return
    stickToBottom.current = el.scrollHeight - el.scrollTop - el.clientHeight < 120
  }

  return (
    <div className="transcript" ref={scrollerRef} onScroll={handleScroll}>
      <div className="transcript__inner">
        {messages.map((m) => (
          <MessageBubble
            key={m.key}
            msg={m}
            dir={dirFor(m.content || '', lang)}
            onUseAuto={onUseAuto}
            onRetry={onRetry}
          />
        ))}
        <div ref={endRef} />
      </div>
    </div>
  )
}
