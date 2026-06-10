import { Link } from 'react-router-dom'
import type { ConversationSummary } from '../api/types'
import { dirFor } from '../i18n'
import { useApp } from '../state/AppContext'
import ModelPicker from './ModelPicker'

interface Props {
  conversations: ConversationSummary[]
  activeId: string | null
  open: boolean
  onClose: () => void
  onSelect: (id: string) => void
  onNew: () => void
  onRename: (id: string) => void
  onDelete: (id: string) => void
}

export default function Sidebar({
  conversations,
  activeId,
  open,
  onClose,
  onSelect,
  onNew,
  onRename,
  onDelete,
}: Props) {
  const { d, lang, toggleLang } = useApp()

  return (
    <>
      <aside className={`sidebar ${open ? 'open' : ''}`}>
        <div className="sidebar__brand">
          <h1>{d.appName}</h1>
          <span className="brand-alt">{d.appNameAlt}</span>
          <span className="brand-rule" />
        </div>

        <button type="button" className="btn-new-chat" onClick={onNew}>
          <span className="plus">✦</span> {d.newChat}
        </button>

        <div className="sidebar__section-label">{d.conversations}</div>
        <nav className="conv-list">
          {conversations.length === 0 && <div className="sidebar__empty">{d.noConversations}</div>}
          {conversations.map((c) => (
            <button
              type="button"
              key={c.id}
              className={`conv-item ${c.id === activeId ? 'active' : ''}`}
              onClick={() => onSelect(c.id)}
            >
              <span className="conv-title" dir={dirFor(c.title || '…', lang)}>
                {c.title || '…'}
              </span>
              <span className="conv-actions">
                <span
                  role="button"
                  tabIndex={0}
                  className="icon-btn"
                  title={d.rename}
                  onClick={(e) => {
                    e.stopPropagation()
                    onRename(c.id)
                  }}
                  onKeyDown={(e) => e.key === 'Enter' && onRename(c.id)}
                >
                  ✎
                </span>
                <span
                  role="button"
                  tabIndex={0}
                  className="icon-btn"
                  title={d.delete}
                  onClick={(e) => {
                    e.stopPropagation()
                    onDelete(c.id)
                  }}
                  onKeyDown={(e) => e.key === 'Enter' && onDelete(c.id)}
                >
                  ✕
                </span>
              </span>
            </button>
          ))}
        </nav>

        <div className="sidebar__footer">
          <ModelPicker />
          <div className="sidebar__footer-row">
            <Link className="nav-link" to="/dashboard">
              <span className="glyph">▦</span> {d.dashboard}
            </Link>
            <button type="button" className="lang-toggle" onClick={toggleLang}>
              {d.langToggle}
            </button>
          </div>
        </div>
      </aside>
      <button type="button" className={`scrim ${open ? 'show' : ''}`} onClick={onClose} aria-label="close" />
    </>
  )
}
