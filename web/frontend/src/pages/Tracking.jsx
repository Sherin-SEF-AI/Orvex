import { useState } from 'react'
import { runTracking, getTrackingResults, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  track_thresh: 0.5,
  track_buffer: 30,
  match_thresh: 0.8,
  frame_rate: 30.0,
  class_filter: [],
}

export default function Tracking({ sessionId }) {
  const [cfg, setCfg]           = useState(DEFAULT_CFG)
  const [classInput, setClassInput] = useState('')
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [summary, setSummary]   = useState(null)
  const [error, setError]       = useState(null)

  if (!sessionId) return <p className="text-muted text-sm">Select a session first.</p>

  async function handleRun() {
    const cfgWithFilter = {
      ...cfg,
      class_filter: classInput.trim()
        ? classInput.split(',').map((s) => s.trim()).filter(Boolean)
        : [],
    }
    setRunning(true); setError(null); setLog([]); setProgress(0); setSummary(null)
    try {
      const { task_id } = await runTracking(sessionId, cfgWithFilter)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getTrackingResults(sessionId).then((d) => setSummary(d.summary)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Multi-Object Tracking (ByteTrack)</h2>

      <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 text-xs text-yellow-300">
        Requires auto-label to be run first. Assigns persistent IDs to detected objects across frames.
      </div>

      <div className="bg-panel border border-accent rounded p-4 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Track threshold
          <input type="range" min={10} max={90}
            value={Math.round(cfg.track_thresh * 100)}
            onChange={(e) => setCfg((c) => ({ ...c, track_thresh: Number(e.target.value) / 100 }))}
            className="w-28"
          />
          <span className="text-gray-400">{cfg.track_thresh.toFixed(2)}</span>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Track buffer (frames)
          <input type="number" min={5} max={60}
            value={cfg.track_buffer}
            onChange={(e) => setCfg((c) => ({ ...c, track_buffer: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Match threshold
          <input type="range" min={50} max={100}
            value={Math.round(cfg.match_thresh * 100)}
            onChange={(e) => setCfg((c) => ({ ...c, match_thresh: Number(e.target.value) / 100 }))}
            className="w-28"
          />
          <span className="text-gray-400">{cfg.match_thresh.toFixed(2)}</span>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Frame rate (fps)
          <input type="number" step={1} min={1} max={120}
            value={cfg.frame_rate}
            onChange={(e) => setCfg((c) => ({ ...c, frame_rate: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted w-64">
          Class filter (comma-separated, empty = all)
          <input type="text" placeholder="car, truck, person"
            value={classInput}
            onChange={(e) => setClassInput(e.target.value)}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          />
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Running…' : 'Run Tracking'}
      </button>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {progress !== null && (
        <div className="bg-panel border border-accent rounded p-1">
          <div className="h-2 bg-bg rounded overflow-hidden">
            <div className="h-2 bg-highlight rounded transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      <ProgressLog lines={log} />

      {summary && (
        <div className="bg-panel border border-accent rounded p-4 text-sm">
          <p className="text-gray-400 font-bold mb-2">Results</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div><span className="text-muted">Unique objects: </span><span>{summary.total_unique_objects}</span></div>
            <div><span className="text-muted">Max simultaneous: </span><span>{summary.max_simultaneous_tracks}</span></div>
            <div><span className="text-muted">Mean per frame: </span><span>{summary.mean_objects_per_frame?.toFixed(1)}</span></div>
            <div><span className="text-muted">Static objects: </span><span>{summary.static_objects_percent?.toFixed(1)}%</span></div>
            <div><span className="text-muted">Mean duration: </span><span>{summary.mean_track_duration_seconds?.toFixed(1)}s</span></div>
            <div><span className="text-muted">Fragmentation: </span><span>{summary.track_fragmentation_rate?.toFixed(3)}</span></div>
          </div>
          {summary.objects_per_class && (
            <div className="mt-3">
              <p className="text-muted text-xs mb-1">Objects by class:</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(summary.objects_per_class).map(([cls, cnt]) => (
                  <span key={cls} className="bg-accent px-2 py-0.5 rounded text-xs text-gray-300">
                    {cls}: {cnt}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
