import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev server proxies API paths to the FastAPI app so the SPA can use
// same-origin relative URLs in both dev and production. SSE streams fine
// through Vite's http-proxy.
const API = 'http://localhost:8000'
const API_PATHS = [
  '/chat',
  '/query',
  '/status',
  '/health',
  '/models',
  '/books',
  '/jobs',
  '/upload',
  '/ingest',
  '/conversations',
]

export default defineConfig({
  plugins: [react()],
  build: {
    // Build straight into the FastAPI static dir; api/main.py serves it at /.
    outDir: '../api/static/dist',
    emptyOutDir: true,
  },
  server: {
    proxy: Object.fromEntries(API_PATHS.map((p) => [p, { target: API }])),
  },
})
