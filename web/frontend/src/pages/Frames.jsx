import { useState, useEffect, useRef } from 'react'

/**
 * Frames page — frame browser with blur/dup heatmap and filter toggles.
 * Frames are served as static files from /frames/<session_id>/cam0/data/*.jpg
 * (requires FastAPI static mount: app.mount("/frames", StaticFiles(...))).
 */
export default function Frames({ sessionId, extractedSession }) {
  const [frames, setFrames]           = useState([])   // list of { path, timestamp_ns, blur_score, is_blurry, is_duplicate }
  const [filterBlurry, setFilterBlurry] = useState(false)
  const [filterDup, setFilterDup]       = useState(false)
  const canvasRef = useRef(null)

  // Accept frame metadata from parent (ExtractedSession.frame_metadata) or empty
  useEffect(() => {
    if (extractedSession?.frame_metadata) {
      setFrames(extractedSession.frame_metadata)
    } else {
      setFrames([])
    }
  }, [extractedSession, sessionId])

  // Draw heatmap whenever frames change
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas || !frames.length) return
    const ctx = canvas.getContext('2d')
    const W = canvas.offsetWidth || 600
    canvas.width = W
    canvas.height = 36
    const N = Math.max(1, Math.floor(W / 4))
    const bucketSize = Math.ceil(frames.length / N)
    for (let b = 0; b < N; b++) {
      const slice = frames.slice(b * bucketSize, (b + 1) * bucketSize)
      if (!slice.length) continue
      const avg = slice.reduce((sum, f) => sum + (f.blur_score ?? 0), 0) / slice.length
      const color = avg > 200 ? '#22c55e' : avg > 100 ? '#eab308' : '#ef4444'
      ctx.fillStyle = color
      ctx.fillRect(Math.round(b * W / N), 0, Math.round(W / N) + 1, 36)
    }
  }, [frames])

  if (!sessionId) return <p className="text-muted text-sm">Select a session.</p>

  const displayed = frames.filter((f) => {
    if (filterBlurry && f.is_blurry) return false
    if (filterDup    && f.is_duplicate) return false
    return true
  })

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-4">
        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide flex-1">Frame Browser</h2>
        <label className="flex items-center gap-2 text-xs text-muted cursor-pointer">
          <input type="checkbox" checked={filterBlurry} onChange={(e) => setFilterBlurry(e.target.checked)}
            className="accent-highlight" />
          Hide blurry
        </label>
        <label className="flex items-center gap-2 text-xs text-muted cursor-pointer">
          <input type="checkbox" checked={filterDup} onChange={(e) => setFilterDup(e.target.checked)}
            className="accent-highlight" />
          Hide duplicates
        </label>
        <span className="text-xs text-muted">{displayed.length} / {frames.length} frames</span>
      </div>

      {/* Blur quality heatmap strip */}
      {frames.length > 0 && (
        <div className="rounded overflow-hidden border border-accent/40">
          <canvas ref={canvasRef} style={{ width: '100%', height: 36, display: 'block' }} />
          <div className="flex justify-between px-1 py-0.5 bg-panel text-xs text-muted">
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 bg-green-500 rounded-sm" />Sharp (&gt;200)</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 bg-yellow-500 rounded-sm" />Moderate</span>
            <span className="flex items-center gap-1"><span className="inline-block w-2 h-2 bg-red-500 rounded-sm" />Blurry (&lt;100)</span>
          </div>
        </div>
      )}

      {/* Frame grid or placeholder */}
      {frames.length === 0 ? (
        <div className="bg-panel border border-accent rounded p-6 text-center text-muted text-sm">
          <p className="mb-2">No frame metadata available for this session.</p>
          <p className="text-xs">Run extraction first to populate frame metadata.</p>
          <p className="mt-3 text-xs">Session ID: <span className="font-mono text-gray-300">{sessionId}</span></p>
        </div>
      ) : (
        <div className="grid grid-cols-4 sm:grid-cols-6 lg:grid-cols-8 gap-2 overflow-y-auto max-h-[60vh]">
          {displayed.map((f, i) => {
            const fname = f.frame_path?.split('/').pop() ?? `frame_${i}`
            const tsMs = f.timestamp_ns != null ? (f.timestamp_ns / 1e6).toFixed(0) : '?'
            const blurLabel = f.blur_score != null ? `blur: ${f.blur_score.toFixed(0)}` : ''
            return (
              <div
                key={i}
                title={`${fname}\nt=${tsMs}ms\n${blurLabel}${f.is_blurry ? ' BLURRY' : ''}${f.is_duplicate ? ' DUP' : ''}`}
                className={`relative bg-panel border rounded overflow-hidden aspect-square flex items-center justify-center
                  ${f.is_blurry ? 'border-red-700' : f.is_duplicate ? 'border-yellow-700' : 'border-accent/40'}`}
              >
                <img
                  src={`/frames/${sessionId}/cam0/data/${fname}`}
                  alt={fname}
                  loading="lazy"
                  className="w-full h-full object-cover"
                  onError={(e) => { e.target.style.display = 'none' }}
                />
                {(f.is_blurry || f.is_duplicate) && (
                  <span className={`absolute bottom-0 right-0 text-[9px] px-1 font-bold
                    ${f.is_blurry ? 'bg-red-900 text-red-300' : 'bg-yellow-900 text-yellow-300'}`}>
                    {f.is_blurry ? 'BLR' : 'DUP'}
                  </span>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
