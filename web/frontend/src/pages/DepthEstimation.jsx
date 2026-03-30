import { useState } from 'react'
import { runDepth, getDepthResults, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  model_variant: 'small',
  batch_size: 8,
  device: 'auto',
  colorize: true,
}

export default function DepthEstimation({ sessionId }) {
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
      const { task_id } = await runDepth(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getDepthResults(sessionId)
              .then((d) => setSummary(d.summary))
              .catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Depth Estimation</h2>

      <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 text-xs text-yellow-300">
        Depth-Anything-v2 produces <strong>relative depth only</strong> — values are not metric distances.
      </div>

      <div className="bg-panel border border-accent rounded p-4 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Model variant
          <select
            value={cfg.model_variant}
            onChange={(e) => setCfg((c) => ({ ...c, model_variant: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {['small','base','large'].map((v) => <option key={v}>{v}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Batch size
          <input type="number" min={1} max={32}
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
        <label className="flex items-center gap-2 text-xs text-muted mt-4">
          <input type="checkbox" checked={cfg.colorize}
            onChange={(e) => setCfg((c) => ({ ...c, colorize: e.target.checked }))}
          />
          Colorize depth (plasma)
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Running…' : 'Estimate Depth'}
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
            <div><span className="text-muted">Avg mean depth: </span><span>{summary.mean_depth_avg?.toFixed(3)}</span></div>
            <div><span className="text-muted">Metric: </span>
              <span className="text-yellow-400">Relative only (not metric)</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
