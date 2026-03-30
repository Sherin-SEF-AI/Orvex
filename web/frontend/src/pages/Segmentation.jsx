import { useState } from 'react'
import { runSegmentation, getSegmentationResults, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  model_name: 'nvidia/segformer-b2-finetuned-cityscapes-512-1024',
  batch_size: 4,
  device: 'auto',
  overlay_alpha: 0.5,
  mark_invalid: true,
}

const MODEL_OPTIONS = [
  { label: 'SegFormer-B0 (fast)',     value: 'nvidia/segformer-b0-finetuned-cityscapes-512-1024' },
  { label: 'SegFormer-B2 (balanced)', value: 'nvidia/segformer-b2-finetuned-cityscapes-512-1024' },
  { label: 'SegFormer-B5 (accurate)', value: 'nvidia/segformer-b5-finetuned-cityscapes-640-1280' },
]

export default function Segmentation({ sessionId }) {
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
      const { task_id } = await runSegmentation(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getSegmentationResults(sessionId).then((d) => setSummary(d.summary)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Semantic Segmentation</h2>

      <div className="bg-panel border border-accent rounded p-4 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Model
          <select
            value={cfg.model_name}
            onChange={(e) => setCfg((c) => ({ ...c, model_name: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {MODEL_OPTIONS.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Batch size
          <input type="number" min={1} max={16}
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
            {['auto', 'cpu', 'cuda:0'].map((d) => <option key={d}>{d}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Overlay opacity
          <input type="range" min={0} max={100}
            value={Math.round(cfg.overlay_alpha * 100)}
            onChange={(e) => setCfg((c) => ({ ...c, overlay_alpha: Number(e.target.value) / 100 }))}
            className="w-28"
          />
          <span className="text-gray-400">{Math.round(cfg.overlay_alpha * 100)}%</span>
        </label>
        <label className="flex items-center gap-2 text-xs text-muted mt-4">
          <input type="checkbox" checked={cfg.mark_invalid}
            onChange={(e) => setCfg((c) => ({ ...c, mark_invalid: e.target.checked }))}
          />
          Flag non-road frames as invalid
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Running…' : 'Run Segmentation'}
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
            <div><span className="text-muted">Frames processed: </span><span>{summary.frame_count}</span></div>
            <div><span className="text-muted">Frames with road: </span><span>{summary.frames_with_road}</span></div>
            <div><span className="text-muted">Invalid frames: </span>
              <span className={summary.invalid_frame_count > 0 ? 'text-yellow-400' : 'text-green-400'}>
                {summary.invalid_frame_count}
              </span>
            </div>
            <div><span className="text-muted">Mean road coverage: </span>
              <span>{summary.mean_road_coverage?.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
