import { useState } from 'react'
import { runLaneDetection, getLaneResults, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  use_ufld: false,
  ufld_conf_threshold: 0.5,
  classical_fallback: true,
  roi_top_percent: 0.55,
  camera_height_m: 1.0,
  camera_pitch_deg: 0.0,
  model_path: null,
}

export default function LaneDetection({ sessionId }) {
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
      const { task_id } = await runLaneDetection(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getLaneResults(sessionId).then((d) => setSummary(d.summary)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  const departureBg = (v) => {
    if (v === undefined) return 'text-gray-400'
    const n = parseFloat(v)
    if (n < 5)  return 'text-green-400'
    if (n < 15) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Lane Detection — Indian Roads</h2>

      <div className="bg-blue-900/20 border border-blue-700 rounded p-3 text-xs text-blue-300">
        Optimized for Indian roads including <strong>unmarked roads</strong>.
        "No lane markings detected" is a valid result — the pipeline will not fail silently.
      </div>

      <div className="bg-panel border border-accent rounded p-4 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          ROI top % (crop from top)
          <input type="range" min={30} max={80}
            value={Math.round(cfg.roi_top_percent * 100)}
            onChange={(e) => setCfg((c) => ({ ...c, roi_top_percent: Number(e.target.value) / 100 }))}
            className="w-28"
          />
          <span className="text-gray-400">{Math.round(cfg.roi_top_percent * 100)}%</span>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Camera height (m)
          <input type="number" step={0.1} min={0.3} max={3.0}
            value={cfg.camera_height_m}
            onChange={(e) => setCfg((c) => ({ ...c, camera_height_m: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Camera pitch (°)
          <input type="number" step={1} min={-30} max={30}
            value={cfg.camera_pitch_deg}
            onChange={(e) => setCfg((c) => ({ ...c, camera_pitch_deg: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-muted mt-4">
          <input type="checkbox" checked={cfg.classical_fallback}
            onChange={(e) => setCfg((c) => ({ ...c, classical_fallback: e.target.checked }))}
          />
          Classical CV fallback
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Running…' : 'Detect Lanes'}
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
            <div><span className="text-muted">Frames with lanes: </span><span>{summary.frames_with_lanes}</span></div>
            <div><span className="text-muted">Frames without lanes: </span>
              <span className="text-yellow-400">{summary.frames_without_lanes}</span>
            </div>
            <div><span className="text-muted">Unmarked road: </span>
              <span className={departureBg(summary.unmarked_road_percent)}>
                {summary.unmarked_road_percent?.toFixed(1)}%
              </span>
            </div>
            <div><span className="text-muted">Mean lateral offset: </span>
              <span>{summary.mean_lateral_offset?.toFixed(3)}</span>
            </div>
            {summary.detection_method_distribution && Object.entries(summary.detection_method_distribution).map(([k, v]) => (
              <div key={k}><span className="text-muted">{k}: </span><span>{v?.toFixed(1)}%</span></div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
