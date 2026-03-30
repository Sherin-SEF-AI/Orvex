import { useState } from 'react'
import { runAutolabel, getAutolabelResults, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'
import StatusBadge from '../components/StatusBadge'

const DEFAULT_CFG = {
  model_path: 'yolov8n.pt',
  conf_threshold: 0.25,
  iou_threshold: 0.45,
  batch_size: 16,
  device: 'auto',
  export_format: 'both',
}

const ROVER_CLASSES = [
  'person','bicycle','car','motorcycle','autorickshaw',
  'bus','truck','traffic light','stop sign','dog','cow',
  'pothole_region','speed_bump',
]

export default function AutoLabel({ sessionId }) {
  const [cfg, setCfg]           = useState(DEFAULT_CFG)
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [stats, setStats]       = useState(null)
  const [error, setError]       = useState(null)

  if (!sessionId) return <p className="text-muted text-sm">Select a session first.</p>

  async function handleRun() {
    setRunning(true); setError(null); setLog([]); setProgress(0); setStats(null)
    try {
      const { task_id } = await runAutolabel(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getAutolabelResults(sessionId)
              .then((d) => setStats(d.stats))
              .catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Auto-Label</h2>

      {/* Config */}
      <div className="bg-panel border border-accent rounded p-4 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Model
          <select
            value={cfg.model_path}
            onChange={(e) => setCfg((c) => ({ ...c, model_path: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {['yolov8n.pt','yolov8s.pt','yolov8m.pt','yolov8l.pt','yolov8x.pt'].map((m) => (
              <option key={m}>{m}</option>
            ))}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Confidence
          <input type="number" min={0.01} max={0.99} step={0.01}
            value={cfg.conf_threshold}
            onChange={(e) => setCfg((c) => ({ ...c, conf_threshold: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          IoU
          <input type="number" min={0.01} max={0.99} step={0.01}
            value={cfg.iou_threshold}
            onChange={(e) => setCfg((c) => ({ ...c, iou_threshold: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Batch size
          <input type="number" min={1} max={64}
            value={cfg.batch_size}
            onChange={(e) => setCfg((c) => ({ ...c, batch_size: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Device
          <select
            value={cfg.device}
            onChange={(e) => setCfg((c) => ({ ...c, device: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {['auto','cpu','cuda:0'].map((d) => <option key={d}>{d}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Export format
          <select
            value={cfg.export_format}
            onChange={(e) => setCfg((c) => ({ ...c, export_format: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {['both','cvat','yolo'].map((f) => <option key={f}>{f}</option>)}
          </select>
        </label>
      </div>

      {/* Classes info */}
      <div className="bg-panel border border-accent rounded p-3">
        <p className="text-xs text-muted mb-1">Rover classes ({ROVER_CLASSES.length})</p>
        <div className="flex flex-wrap gap-1">
          {ROVER_CLASSES.map((c) => (
            <span key={c} className="text-xs bg-accent text-gray-300 px-2 py-0.5 rounded">{c}</span>
          ))}
        </div>
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Running…' : 'Run Auto-Label'}
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

      {stats && (
        <div className="bg-panel border border-accent rounded p-4 text-sm">
          <p className="text-gray-400 font-bold mb-2">Results</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <Stat label="Frames processed" value={stats.total_frames} />
            <Stat label="Total detections" value={stats.total_detections} />
            <Stat label="Avg detections/frame" value={stats.avg_detections_per_frame?.toFixed(2)} />
            <Stat label="Empty frames" value={stats.empty_frames} />
          </div>
        </div>
      )}
    </div>
  )
}

function Stat({ label, value }) {
  return (
    <div>
      <span className="text-muted">{label}: </span>
      <span className="text-gray-200">{value ?? '—'}</span>
    </div>
  )
}
