import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import { del, get } from '../api/client'
import type { BookRow, BooksResponse, JobInfo } from '../api/types'
import BooksTable from '../components/dashboard/BooksTable'
import JobProgress from '../components/dashboard/JobProgress'
import UploadCard from '../components/dashboard/UploadCard'
import { usePolling } from '../hooks/usePolling'
import { useApp } from '../state/AppContext'

export default function DashboardPage() {
  const { d, toggleLang } = useApp()
  const [books, setBooks] = useState<BookRow[]>([])
  const [totals, setTotals] = useState({ books: 0, chunks: 0 })
  const [jobs, setJobs] = useState<JobInfo[]>([])

  const refreshBooks = useCallback(async () => {
    const res = await get<BooksResponse>('/books')
    setBooks(res.books)
    setTotals({ books: res.total_books, chunks: res.total_chunks })
  }, [])

  const refreshJobs = useCallback(async () => {
    const res = await get<{ jobs: JobInfo[] }>('/jobs')
    setJobs((prev) => {
      // when a job transitions to finished/failed, refresh the books table
      const prevActive = new Set(
        prev.filter((j) => j.state === 'queued' || j.state === 'started').map((j) => j.job_id),
      )
      const nowDone = res.jobs.some(
        (j) => prevActive.has(j.job_id) && (j.state === 'finished' || j.state === 'failed'),
      )
      if (nowDone) void refreshBooks()
      return res.jobs
    })
  }, [refreshBooks])

  usePolling(refreshBooks, 15000)
  const anyActive = jobs.some((j) => j.state === 'queued' || j.state === 'started')
  usePolling(refreshJobs, anyActive ? 2000 : 6000)

  const deleteBook = useCallback(
    (book: BookRow) => {
      if (!window.confirm(d.confirmDeleteBook(book.title))) return
      del(`/books/${book.book_id}`)
        .then(refreshBooks)
        .catch((e: unknown) => window.alert(`${d.error}: ${e instanceof Error ? e.message : e}`))
    },
    [d, refreshBooks],
  )

  const activeJobs = jobs.filter((j) => j.state === 'queued' || j.state === 'started').length

  return (
    <div className="app">
      <div className="dash">
        <div className="dash__inner">
          <div className="dash__head">
            <h2>{d.dashboard}</h2>
            <span className="rule" />
            <Link className="scope-button" to="/" style={{ textDecoration: 'none' }}>
              ← {d.backToChat}
            </Link>
            <button type="button" className="scope-button" onClick={toggleLang}>
              {d.langToggle}
            </button>
          </div>
          <p className="dash__sub">{d.tagline}</p>

          <div className="stat-row">
            <div className="stat-card" style={{ ['--accent-bar' as never]: 'var(--gold)' }}>
              <div className="stat-value">{totals.books}</div>
              <div className="stat-label">{d.totalBooks}</div>
            </div>
            <div className="stat-card" style={{ ['--accent-bar' as never]: 'var(--madder)' }}>
              <div className="stat-value">{totals.chunks.toLocaleString()}</div>
              <div className="stat-label">{d.totalChunks}</div>
            </div>
            <div className="stat-card" style={{ ['--accent-bar' as never]: 'var(--sage)' }}>
              <div className="stat-value">{activeJobs}</div>
              <div className="stat-label">{d.activeJobs}</div>
            </div>
          </div>

          <UploadCard onQueued={refreshJobs} />
          <JobProgress jobs={jobs} />
          <BooksTable books={books} onDelete={deleteBook} />
        </div>
      </div>
    </div>
  )
}
