import { useEffect, useRef } from 'react'
import { useApp } from '../state/AppContext'

interface Props {
  value: string
  onChange: (v: string) => void
  onSend: () => void
  onStop: () => void
  busy: boolean
}

export default function Composer({ value, onChange, onSend, onStop, busy }: Props) {
  const { d } = useApp()
  const ref = useRef<HTMLTextAreaElement>(null)

  // autosize
  useEffect(() => {
    const el = ref.current
    if (!el) return
    el.style.height = 'auto'
    el.style.height = `${Math.min(el.scrollHeight, 180)}px`
  }, [value])

  const handleKey = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault()
      if (!busy && value.trim()) onSend()
    }
  }

  return (
    <div className="composer">
      <div className="composer__box">
        <textarea
          ref={ref}
          rows={1}
          value={value}
          placeholder={d.askPlaceholder}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKey}
          dir="auto"
        />
        {busy ? (
          <button type="button" className="btn-send stop" onClick={onStop} title={d.stop}>
            <svg width="13" height="13" viewBox="0 0 13 13" aria-hidden="true">
              <rect width="13" height="13" rx="2.5" fill="currentColor" />
            </svg>
          </button>
        ) : (
          <button
            type="button"
            className="btn-send"
            onClick={onSend}
            disabled={!value.trim()}
            title={d.send}
          >
            <svg className="arrow" width="17" height="17" viewBox="0 0 17 17" aria-hidden="true">
              <path
                d="M2 8.5h11M9.5 4.5l4 4-4 4"
                stroke="currentColor"
                strokeWidth="1.8"
                fill="none"
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            </svg>
          </button>
        )}
      </div>
      <div className="composer__hint">Enter ↵ · Shift+Enter ⏎</div>
    </div>
  )
}
