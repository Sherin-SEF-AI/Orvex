import { useState } from 'react'
import { runOccupancy, getOccupancyResults, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  grid_resolution_m: 0.1,
  grid_width_m: 20.0,
  grid_height_m: 30.0,
  camera_height_m: 1.0,
  max_depth_m: 30.0,
  temporal_fusion_window: 5,
  decay_factor: 0.95,
}

export default function OccupancyGrid({ sessionId }) {
  const [cfg, setCfg]           = useState(DEFAULT_CFG)
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [summary, setSummary]   = useState(null)
  const [error, setError]       = useState(null)

  if (!sessionId) return <p className="text-muted text-sm">Select a session first.</p>

  async function handleRun() {
    setRunning(true); setError(null); setLog([]); setProgress(0); setSummary(null)
    try {
      const { task_id } = await runOccupancy(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getOccupancyResults(sessionId).then((d) => setSummary(d.summary)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  function Field({ label, field, type = 'number', step, min, max }) {
    return (
      <label className="flex flex-col gap-1 text-xs text-muted">
        {label}
        <input type={type} step={step} min={min} max={max}
          value={cfg[field]}
          onChange={(e) => setCfg((c) => ({ ...c, [field]: Number(e.target.value) }))}
          className="bg-bg border border-accent rounded px-2 py-1 text-sm w-24 text-gray-300"
        />
      </label>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Occupancy Grid</h2>

      <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 text-xs text-yellow-300">
        Requires depth estimation and auto-label to be run first on this session.
        If camera intrinsics not calibrated, uses 1.0m camera height estimate.
      </div>

      <div className="bg-panel border border-accent rounded p-4 flex flex-wrap gap-4">
        <Field label="Grid resolution (m)" field="grid_resolution_m" step={0.05} min={0.05} max={0.5} />
        <Field label="Grid width (m)"      field="grid_width_m"      step={1}    min={5}    max={50}  />
        <Field label="Grid height (m)"     field="grid_height_m"     step={1}    min={5}    max={60}  />
        <Field label="Camera height (m)"   field="camera_height_m"   step={0.1}  min={0.3}  max={3.0} />
        <Field label="Max depth (m)"       field="max_depth_m"       step={1}    min={5}    max={100} />
        <Field label="Fusion window"       field="temporal_fusion_window" step={1} min={1} max={20} />
        <Field label="Decay factor"        field="decay_factor"      step={0.05} min={0.5}  max={1.0} />
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Running…' : 'Generate Occupancy Grid'}
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
            <div><span className="text-muted">Frames: </span><span>{summary.frame_count}</span></div>
            <div><span className="text-muted">Mean occupancy: </span><span>{summary.mean_occupancy_percent?.toFixed(1)}%</span></div>
            <div><span className="text-muted">High density frames: </span>
              <span className={summary.high_density_frames > 0 ? 'text-yellow-400' : 'text-green-400'}>
                {summary.high_density_frames}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
