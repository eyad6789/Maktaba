import { useRef, useState } from 'react'
import { uploadPdf } from '../../api/client'
import type { UploadResponse } from '../../api/types'
import { useApp } from '../../state/AppContext'

interface ActiveUpload {
  name: string
  percent: number
  state: 'uploading' | 'queued' | 'duplicate' | 'error'
  note?: string
}

interface Props {
  onQueued: () => void
}

export default function UploadCard({ onQueued }: Props) {
  const { d } = useApp()
  const [title, setTitle] = useState('')
  const [author, setAuthor] = useState('')
  const [over, setOver] = useState(false)
  const [uploads, setUploads] = useState<ActiveUpload[]>([])
  const fileRef = useRef<HTMLInputElement>(null)

  const patchUpload = (name: string, p: Partial<ActiveUpload>) =>
    setUploads((prev) => prev.map((u) => (u.name === name ? { ...u, ...p } : u)))

  const handleFiles = (files: FileList | File[]) => {
    for (const file of Array.from(files)) {
      if (!file.name.toLowerCase().endsWith('.pdf')) continue
      setUploads((prev) => [...prev, { name: file.name, percent: 0, state: 'uploading' }])
      uploadPdf(file, { title: title.trim() || undefined, author: author.trim() || undefined }, (pct) =>
        patchUpload(file.name, { percent: pct }),
      )
        .then(({ status, body }) => {
          if (status >= 200 && status < 300) {
            const res = JSON.parse(body) as UploadResponse
            if (res.status === 'duplicate') {
              patchUpload(file.name, { state: 'duplicate', note: d.duplicateBook })
            } else {
              patchUpload(file.name, { state: 'queued', percent: 100 })
              onQueued()
            }
          } else {
            let note = `HTTP ${status}`
            try {
              note = JSON.parse(body)?.detail ?? note
            } catch {
              /* keep default */
            }
            patchUpload(file.name, { state: 'error', note: String(note) })
          }
        })
        .catch(() => patchUpload(file.name, { state: 'error', note: 'upload failed' }))
    }
    setTitle('')
    setAuthor('')
  }

  return (
    <section className="panel">
      <h3>{d.uploadTitle}</h3>
      <p className="panel-sub">{d.tagline}</p>

      <div className="upload-fields">
        <input
          value={title}
          onChange={(e) => setTitle(e.target.value)}
          placeholder={d.titleOptional}
          dir="auto"
        />
        <input
          value={author}
          onChange={(e) => setAuthor(e.target.value)}
          placeholder={d.authorOptional}
          dir="auto"
        />
      </div>

      <div
        className={`dropzone ${over ? 'over' : ''}`}
        onClick={() => fileRef.current?.click()}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          handleFiles(e.dataTransfer.files)
        }}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => e.key === 'Enter' && fileRef.current?.click()}
      >
        <span className="glyph">⇪</span>
        {d.uploadHint}
        <input
          ref={fileRef}
          type="file"
          accept="application/pdf,.pdf"
          multiple
          hidden
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files)
            e.target.value = ''
          }}
        />
      </div>

      {uploads.map((u) => (
        <div className="job-row" key={u.name}>
          <span className="job-title" dir="auto">
            {u.name}
          </span>
          {u.state === 'uploading' && (
            <>
              <span className="job-stage">
                {d.uploading} {u.percent}%
              </span>
              <span className="bar">
                <i style={{ width: `${u.percent}%` }} />
              </span>
            </>
          )}
          {u.state === 'queued' && <span className="state-pill queued">{d.jobQueued}</span>}
          {u.state === 'duplicate' && <span className="state-pill finished">{u.note}</span>}
          {u.state === 'error' && <span className="state-pill failed">{u.note}</span>}
        </div>
      ))}
    </section>
  )
}
