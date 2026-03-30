import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Proxy target for the FastAPI backend
const API = 'http://localhost:8001'

// All API route prefixes — these get proxied to the backend
// BUT only for non-HTML requests (XHR/fetch), not browser navigation
const API_PREFIXES = [
  '/sessions', '/audit', '/extraction', '/calibration', '/dataset',
  '/health', '/autolabel', '/depth', '/slam', '/reconstruction',
  '/active-learning', '/analytics', '/augmentation', '/training',
  '/inference', '/review', '/learning', '/segmentation', '/occupancy',
  '/lanes', '/tracking', '/versioning', '/experiments', '/edge-export',
  '/insta360',
]

// Build proxy entries: skip proxying when browser requests HTML (page navigation)
const proxy = {}
for (const prefix of API_PREFIXES) {
  proxy[prefix] = {
    target: API,
    bypass(req) {
      if (req.headers.accept && req.headers.accept.includes('text/html')) {
        return req.url   // serve Vite's index.html instead of proxying
      }
    },
  }
}

// WebSocket proxy for live task progress
proxy['/ws'] = {
  target: 'ws://localhost:8001',
  ws: true,
  rewriteWsOrigin: true,
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy,
  },
})
