import type { JobInfo } from '../../api/types'
import { jobStateLabel, stageLabel } from '../../i18n'
import { useApp } from '../../state/AppContext'

interface Props {
  jobs: JobInfo[]
}

/** Live vectorization queue: one row per RQ job with stage + progress. */
export default function JobProgress({ jobs }: Props) {
  const { d } = useApp()
  if (!jobs.length) return null

  return (
    <section className="panel">
      <h3>{d.jobs}</h3>
      <p className="panel-sub">
        {d.activeJobs}: {jobs.filter((j) => j.state === 'queued' || j.state === 'started').length}
      </p>
      {jobs.map((j) => {
        const active = j.state === 'queued' || j.state === 'started'
        const pct =
          j.total && j.total > 0 && j.current !== null
            ? Math.min(100, Math.round((j.current / j.total) * 100))
            : null
        return (
          <div className="job-row" key={j.job_id}>
            <span className="job-title" dir="auto">
              {j.title || j.book_id || j.job_id}
            </span>
            {active && (
              <>
                <span className="job-stage">
                  {stageLabel(d, j.stage) ?? jobStateLabel(d, j.state)}
                  {pct !== null ? ` · ${pct}%` : ''}
                </span>
                <span className={`bar ${pct === null ? 'indeterminate' : ''}`}>
                  <i style={pct !== null ? { width: `${pct}%` } : undefined} />
                </span>
              </>
            )}
            {j.state === 'failed' && j.error && (
              <span className="job-stage" title={j.error}>
                {j.error.slice(0, 60)}
              </span>
            )}
            <span className={`state-pill ${j.state}`}>{jobStateLabel(d, j.state)}</span>
          </div>
        )
      })}
    </section>
  )
}
