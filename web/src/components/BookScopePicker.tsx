import { useEffect, useRef, useState } from 'react'
import { get } from '../api/client'
import type { BookRow, BooksResponse } from '../api/types'
import { dirFor } from '../i18n'
import { useApp } from '../state/AppContext'

interface Props {
  selected: string[] | null // null = all books
  onChange: (ids: string[] | null) => void
}

/** Chat-header picker: ask all books, one book, or any subset. */
export default function BookScopePicker({ selected, onChange }: Props) {
  const { d, lang } = useApp()
  const [books, setBooks] = useState<BookRow[]>([])
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    get<BooksResponse>('/books')
      .then((res) => setBooks(res.books.filter((b) => b.status === 'completed')))
      .catch(() => setBooks([]))
  }, [open]) // refresh whenever the popover opens

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  const toggle = (id: string) => {
    if (selected === null) {
      onChange([id])
      return
    }
    const next = selected.includes(id) ? selected.filter((x) => x !== id) : [...selected, id]
    onChange(next.length === 0 ? null : next)
  }

  const label =
    selected === null
      ? d.allBooks
      : selected.length === 1
        ? (books.find((b) => b.book_id === selected[0])?.title ?? d.selectedBooks(1))
        : d.selectedBooks(selected.length)

  return (
    <div className="scope-picker" ref={rootRef}>
      <button type="button" className="scope-button" onClick={() => setOpen((v) => !v)}>
        <span className="glyph">⌘</span>
        <span style={{ maxWidth: 200, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
          {label}
        </span>
        <span className="caret">▾</span>
      </button>

      {open && (
        <div className="popover">
          <label className="check-row">
            <input
              type="checkbox"
              checked={selected === null}
              onChange={() => onChange(null)}
            />
            <span>{d.allBooks}</span>
            <span className="row-sub">{books.length}</span>
          </label>
          <div className="popover__sep" />
          {books.map((b) => (
            <label className="check-row" key={b.book_id} dir={dirFor(b.title, lang)}>
              <input
                type="checkbox"
                checked={selected !== null && selected.includes(b.book_id)}
                onChange={() => toggle(b.book_id)}
              />
              <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{b.title}</span>
              {b.author && <span className="row-sub">{b.author}</span>}
            </label>
          ))}
          {books.length === 0 && <div className="empty-note">{d.noBooks}</div>}
        </div>
      )}
    </div>
  )
}
