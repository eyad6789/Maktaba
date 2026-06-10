import { useEffect, useRef, useState } from 'react'
import { get } from '../api/client'
import type { ModelsResponse, ProviderInfo } from '../api/types'
import { useApp } from '../state/AppContext'

/** Sidebar model selector: Auto + every provider in the fallback chain. */
export default function ModelPicker() {
  const { d, provider, setProvider } = useApp()
  const [providers, setProviders] = useState<ProviderInfo[]>([])
  const [open, setOpen] = useState(false)
  const rootRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    get<ModelsResponse>('/models')
      .then((res) => setProviders(res.providers))
      .catch(() => setProviders([]))
  }, [])

  useEffect(() => {
    if (!open) return
    const close = (e: MouseEvent) => {
      if (!rootRef.current?.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', close)
    return () => document.removeEventListener('mousedown', close)
  }, [open])

  const current = providers.find((p) => p.id === provider)
  const label = provider === 'auto' ? d.modelAuto : (current?.label ?? provider)

  return (
    <div className="model-picker" ref={rootRef}>
      <button
        type="button"
        className="model-picker__button"
        onClick={() => setOpen((v) => !v)}
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className={`dot ${provider === 'auto' ? 'dot--auto' : current?.available ? 'dot--ok' : 'dot--off'}`} />
        <span>{label}</span>
        {provider === 'auto' && <span className="model-picker__hint">{d.modelAutoHint}</span>}
        <span className="caret">▾</span>
      </button>

      {open && (
        <div className="popover" role="listbox">
          {providers.map((p, i) => (
            <div key={p.id}>
              {i === 1 && <div className="popover__sep" />}
              <button
                type="button"
                role="option"
                aria-selected={provider === p.id}
                className={`popover__item ${provider === p.id ? 'selected' : ''}`}
                disabled={!p.available}
                onClick={() => {
                  setProvider(p.id)
                  setOpen(false)
                }}
              >
                <span className={`dot ${p.id === 'auto' ? 'dot--auto' : p.available ? 'dot--ok' : 'dot--off'}`} />
                <span>
                  {p.id === 'auto' ? d.modelAuto : p.label}
                  <span className="item-sub">
                    {p.id === 'auto' ? d.modelAutoHint : (p.model ?? '') + (p.available ? '' : ` · ${d.unavailable}`)}
                  </span>
                </span>
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
