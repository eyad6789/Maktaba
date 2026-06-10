import { useCallback, useEffect, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router-dom'
import { del, get, patch } from '../api/client'
import { streamChat } from '../api/sse'
import type { ConversationDetail, ConversationSummary } from '../api/types'
import Composer from '../components/Composer'
import MessageList, { type UiMessage } from '../components/MessageList'
import Sidebar from '../components/Sidebar'
import BookScopePicker from '../components/BookScopePicker'
import { useApp } from '../state/AppContext'

let keyCounter = 0
const nextKey = () => `m${++keyCounter}`

export default function ChatPage() {
  const { d, provider, setProvider } = useApp()
  const navigate = useNavigate()
  const params = useParams<{ conversationId?: string }>()

  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<UiMessage[]>([])
  const [input, setInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [scope, setScope] = useState<string[] | null>(null)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [bookCount, setBookCount] = useState(0)

  const abortRef = useRef<AbortController | null>(null)
  const lastSentRef = useRef<string>('')
  const activeIdRef = useRef<string | null>(null)
  activeIdRef.current = activeId

  const refreshConversations = useCallback(() => {
    get<{ conversations: ConversationSummary[] }>('/conversations')
      .then((res) => setConversations(res.conversations))
      .catch(() => setConversations([]))
  }, [])

  useEffect(() => {
    refreshConversations()
    get<{ books: number; chunks: number }>('/status')
      .then((s) => setBookCount(s.books))
      .catch(() => {})
  }, [refreshConversations])

  // Load the conversation named in the URL (skip when we set it ourselves
  // mid-stream — in that case activeId already matches the param).
  useEffect(() => {
    const id = params.conversationId ?? null
    if (id === activeIdRef.current) return
    setActiveId(id)
    if (!id) {
      setMessages([])
      setScope(null)
      return
    }
    get<ConversationDetail>(`/conversations/${id}`)
      .then((conv) => {
        setScope(conv.book_ids)
        setMessages(
          conv.messages.map((m) => ({
            key: nextKey(),
            role: m.role,
            content: m.content,
            citations: m.citations ?? undefined,
            provider: m.model,
            grounded: m.grounded,
          })),
        )
      })
      .catch(() => {
        setMessages([])
        navigate('/', { replace: true })
      })
  }, [params.conversationId, navigate])

  const updateLast = useCallback((patchMsg: Partial<UiMessage>) => {
    setMessages((prev) => {
      if (!prev.length) return prev
      const next = prev.slice()
      next[next.length - 1] = { ...next[next.length - 1], ...patchMsg }
      return next
    })
  }, [])

  const send = useCallback(
    (text: string, { echoUser = true }: { echoUser?: boolean } = {}) => {
      const message = text.trim()
      if (!message || busy) return
      lastSentRef.current = message
      setInput('')
      setBusy(true)
      setMessages((prev) => [
        ...(echoUser ? [...prev, { key: nextKey(), role: 'user' as const, content: message }] : prev),
        { key: nextKey(), role: 'assistant' as const, content: '', streaming: true, searching: true },
      ])

      const controller = new AbortController()
      abortRef.current = controller
      let gotText = false

      streamChat(
        {
          conversation_id: activeIdRef.current,
          message,
          book_ids: scope,
          provider,
        },
        {
          onMeta: (meta) => {
            updateLast({ searching: false, sources: meta.sources })
            if (!activeIdRef.current) {
              activeIdRef.current = meta.conversation_id
              setActiveId(meta.conversation_id)
              navigate(`/c/${meta.conversation_id}`, { replace: true })
              refreshConversations()
            }
          },
          onProvider: (p) => updateLast({ provider: p.provider, model: p.model }),
          onDelta: (text) => {
            gotText = true
            setMessages((prev) => {
              if (!prev.length) return prev
              const next = prev.slice()
              const last = next[next.length - 1]
              next[next.length - 1] = {
                ...last,
                searching: false,
                content: last.content + text,
              }
              return next
            })
          },
          onDone: (done) => {
            updateLast({
              streaming: false,
              searching: false,
              citations: done.citations,
              provider: done.provider,
              model: done.model,
              grounded: done.grounded,
            })
            refreshConversations()
          },
          onError: (err) => {
            updateLast({ streaming: false, searching: false, error: err })
          },
        },
        controller.signal,
      )
        .catch((e: unknown) => {
          const aborted = controller.signal.aborted
          updateLast({
            streaming: false,
            searching: false,
            error: aborted
              ? { provider: null, reason: 'stopped', message: d.stopped, partial: gotText }
              : {
                  provider: null,
                  reason: 'error',
                  message: e instanceof Error ? e.message : String(e),
                  partial: gotText,
                },
          })
        })
        .finally(() => {
          setBusy(false)
          abortRef.current = null
        })
    },
    [busy, scope, provider, navigate, refreshConversations, updateLast, d.stopped],
  )

  const stop = useCallback(() => abortRef.current?.abort(), [])

  const retry = useCallback(() => {
    if (!lastSentRef.current || busy) return
    // Drop the failed assistant bubble, then resend the same text.
    setMessages((prev) => (prev.length && prev[prev.length - 1].error ? prev.slice(0, -1) : prev))
    send(lastSentRef.current, { echoUser: false })
  }, [busy, send])

  const useAuto = useCallback(() => {
    setProvider('auto')
    retry()
  }, [setProvider, retry])

  const newChat = useCallback(() => {
    if (busy) stop()
    activeIdRef.current = null
    setActiveId(null)
    setMessages([])
    setScope(null)
    setSidebarOpen(false)
    navigate('/')
  }, [busy, stop, navigate])

  const selectConversation = useCallback(
    (id: string) => {
      if (busy) stop()
      setSidebarOpen(false)
      navigate(`/c/${id}`)
    },
    [busy, stop, navigate],
  )

  const renameConversation = useCallback(
    (id: string) => {
      const current = conversations.find((c) => c.id === id)
      const title = window.prompt(d.renamePrompt, current?.title ?? '')
      if (title === null) return
      patch(`/conversations/${id}`, { title: title.trim() })
        .then(refreshConversations)
        .catch(() => {})
    },
    [conversations, d.renamePrompt, refreshConversations],
  )

  const deleteConversation = useCallback(
    (id: string) => {
      if (!window.confirm(d.confirmDeleteConv)) return
      del(`/conversations/${id}`)
        .then(() => {
          refreshConversations()
          if (id === activeIdRef.current) newChat()
        })
        .catch(() => {})
    },
    [d.confirmDeleteConv, refreshConversations, newChat],
  )

  const activeConv = conversations.find((c) => c.id === activeId)

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        activeId={activeId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onSelect={selectConversation}
        onNew={newChat}
        onRename={renameConversation}
        onDelete={deleteConversation}
      />

      <main className="chat">
        <header className="chat__header">
          <button
            type="button"
            className="sidebar-toggle"
            onClick={() => setSidebarOpen(true)}
            aria-label="menu"
          >
            ☰
          </button>
          <div className="conv-heading">{activeConv?.title || d.tagline}</div>
          <BookScopePicker selected={scope} onChange={setScope} />
        </header>

        {messages.length === 0 ? (
          <div className="transcript">
            <div className="welcome">
              <div className="ornament">❖ ❖ ❖</div>
              <h2>{d.welcomeTitle}</h2>
              <p className="sub">{d.welcomeSub(bookCount)}</p>
              <div className="example-chips">
                {d.examples.map((ex) => (
                  <button type="button" key={ex} className="chip" dir="auto" onClick={() => setInput(ex)}>
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          <MessageList messages={messages} onUseAuto={useAuto} onRetry={retry} />
        )}

        <Composer value={input} onChange={setInput} onSend={() => send(input)} onStop={stop} busy={busy} />
      </main>
    </div>
  )
}
