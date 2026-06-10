// Fetch wrapper with X-API-Key handling (parity with the legacy static UI):
// the key is bootstrapped from ?api_key=... (then scrubbed from the URL),
// persisted to localStorage, prompted for on 401, and sent on every call.

const KEY_STORAGE = 'ragApiKey'

export class ApiError extends Error {
  status: number
  body: string

  constructor(status: number, body: string) {
    super(`API error ${status}`)
    this.status = status
    this.body = body
  }

  detail(): string {
    try {
      const parsed = JSON.parse(this.body)
      const d = parsed?.detail
      if (typeof d === 'string') return d
      if (d?.message) return String(d.message)
      return this.body
    } catch {
      return this.body
    }
  }
}

function bootstrapKeyFromUrl(): void {
  try {
    const url = new URL(window.location.href)
    const key = url.searchParams.get('api_key')
    if (key) {
      localStorage.setItem(KEY_STORAGE, key)
      url.searchParams.delete('api_key')
      window.history.replaceState({}, '', url.toString())
    }
  } catch {
    /* ignore */
  }
}
bootstrapKeyFromUrl()

export function getApiKey(): string | null {
  try {
    return localStorage.getItem(KEY_STORAGE)
  } catch {
    return null
  }
}

function setApiKey(key: string): void {
  try {
    localStorage.setItem(KEY_STORAGE, key)
  } catch {
    /* ignore */
  }
}

function clearApiKey(): void {
  try {
    localStorage.removeItem(KEY_STORAGE)
  } catch {
    /* ignore */
  }
}

export function apiKeyHeaders(): Record<string, string> {
  const key = getApiKey()
  return key ? { 'X-API-Key': key } : {}
}

/** fetch() with the API key attached; on 401, prompt once and retry. */
export async function authedFetch(path: string, init: RequestInit = {}): Promise<Response> {
  const doFetch = () =>
    fetch(path, {
      ...init,
      headers: { ...(init.headers || {}), ...apiKeyHeaders() },
    })

  let res = await doFetch()
  if (res.status === 401) {
    clearApiKey()
    const entered = window.prompt('API key / مفتاح الواجهة:')
    if (entered && entered.trim()) {
      setApiKey(entered.trim())
      res = await doFetch()
    }
  }
  return res
}

async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const res = await authedFetch(path, init)
  if (!res.ok) throw new ApiError(res.status, await res.text())
  return (await res.json()) as T
}

export const get = <T>(path: string) => api<T>(path)

export const post = <T>(path: string, body?: unknown) =>
  api<T>(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
  })

export const patch = <T>(path: string, body: unknown) =>
  api<T>(path, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })

export const del = <T>(path: string) => api<T>(path, { method: 'DELETE' })

/** Multipart upload via XHR so we get real progress events. */
export function uploadPdf(
  file: File,
  meta: { title?: string; author?: string },
  onProgress: (percent: number) => void,
): Promise<{ status: number; body: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest()
    xhr.open('POST', '/upload')
    const key = getApiKey()
    if (key) xhr.setRequestHeader('X-API-Key', key)
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable) onProgress(Math.round((e.loaded / e.total) * 100))
    }
    xhr.onload = () => resolve({ status: xhr.status, body: xhr.responseText })
    xhr.onerror = () => reject(new Error('upload failed'))
    const form = new FormData()
    form.append('file', file)
    if (meta.title) form.append('title', meta.title)
    if (meta.author) form.append('author', meta.author)
    xhr.send(form)
  })
}
