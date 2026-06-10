import { useEffect, useRef } from 'react'

/**
 * Call `fn` immediately and then every `intervalMs` while `active`.
 * The latest `fn` is always used (no stale closures); overlapping runs are
 * skipped if the previous tick is still in flight.
 */
export function usePolling(fn: () => Promise<void> | void, intervalMs: number, active = true): void {
  const fnRef = useRef(fn)
  fnRef.current = fn

  useEffect(() => {
    if (!active) return
    let stopped = false
    let busy = false
    const tick = async () => {
      if (busy || stopped) return
      busy = true
      try {
        await fnRef.current()
      } catch {
        /* polling never throws to the UI */
      } finally {
        busy = false
      }
    }
    void tick()
    const id = window.setInterval(tick, intervalMs)
    return () => {
      stopped = true
      window.clearInterval(id)
    }
  }, [intervalMs, active])
}
