import type { BookRow } from '../../api/types'
import { dirFor } from '../../i18n'
import { useApp } from '../../state/AppContext'

interface Props {
  books: BookRow[]
  onDelete: (book: BookRow) => void
}

export default function BooksTable({ books, onDelete }: Props) {
  const { d, lang } = useApp()

  return (
    <section className="panel">
      <h3>{d.library}</h3>
      <p className="panel-sub">
        {d.totalBooks}: {books.length}
      </p>
      {books.length === 0 ? (
        <div className="empty-note">{d.noBooks}</div>
      ) : (
        <div style={{ overflowX: 'auto' }}>
          <table className="books-table">
            <thead>
              <tr>
                <th>{d.bookTitle}</th>
                <th>{d.author}</th>
                <th>{d.language}</th>
                <th>{d.pages}</th>
                <th>{d.chunks}</th>
                <th>{d.status}</th>
                <th>{d.updated}</th>
                <th aria-label="actions" />
              </tr>
            </thead>
            <tbody>
              {books.map((b) => (
                <tr key={b.book_id}>
                  <td className="b-title" dir={dirFor(b.title, lang)}>
                    {b.title}
                  </td>
                  <td dir="auto">{b.author || <span className="muted">—</span>}</td>
                  <td>{b.language || <span className="muted">—</span>}</td>
                  <td>{b.num_pages || <span className="muted">—</span>}</td>
                  <td>{b.num_chunks || <span className="muted">—</span>}</td>
                  <td>
                    <span className={`state-pill ${b.status}`} title={b.error ?? undefined}>
                      {b.status}
                    </span>
                  </td>
                  <td className="muted">{b.updated_at ? b.updated_at.slice(0, 10) : '—'}</td>
                  <td>
                    <button type="button" className="btn-danger" onClick={() => onDelete(b)}>
                      {d.delete}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
