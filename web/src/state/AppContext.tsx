import { createContext, useCallback, useContext, useMemo, useState } from 'react'
import type { ReactNode } from 'react'
import { getDict, type Dict, type Lang } from '../i18n'

interface AppState {
  lang: Lang
  d: Dict
  toggleLang: () => void
  /** Selected provider id for new turns: "auto" | "gemini" | "claude" | "local" */
  provider: string
  setProvider: (id: string) => void
}

const Ctx = createContext<AppState | null>(null)

function readLang(): Lang {
  try {
    return localStorage.getItem('maktabah.lang') === 'ar' ? 'ar' : 'en'
  } catch {
    return 'en'
  }
}

function readProvider(): string {
  try {
    return localStorage.getItem('maktabah.provider') || 'auto'
  } catch {
    return 'auto'
  }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(readLang)
  const [provider, setProviderState] = useState<string>(readProvider)

  const toggleLang = useCallback(() => {
    setLang((prev) => {
      const next: Lang = prev === 'en' ? 'ar' : 'en'
      try {
        localStorage.setItem('maktabah.lang', next)
      } catch {
        /* ignore */
      }
      document.documentElement.lang = next
      document.documentElement.dir = next === 'ar' ? 'rtl' : 'ltr'
      return next
    })
  }, [])

  const setProvider = useCallback((id: string) => {
    setProviderState(id)
    try {
      localStorage.setItem('maktabah.provider', id)
    } catch {
      /* ignore */
    }
  }, [])

  const value = useMemo(
    () => ({ lang, d: getDict(lang), toggleLang, provider, setProvider }),
    [lang, toggleLang, provider, setProvider],
  )
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useApp(): AppState {
  const ctx = useContext(Ctx)
  if (!ctx) throw new Error('useApp outside AppProvider')
  return ctx
}
