import { useState } from 'react'
import { runActiveLearning, getActiveLearningResults, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  method: 'entropy',
  n_frames: 100,
  uncertainty_weight: 0.6,
  diversity_weight: 0.4,
}

export default function ActiveLearning({ sessionId }) {
  const [cfg, setCfg]           = useState(DEFAULT_CFG)
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)

  if (!sessionId) return <p className="text-muted text-sm">Select a session first.</p>

  async function handleRun() {
    setRunning(true); setError(null); setLog([]); setProgress(0); setResult(null)
    try {
      const { task_id } = await runActiveLearning(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getActiveLearningResults(sessionId).then((d) => setResult(d.result)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  const budget = result?.budget

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Active Learning</h2>

      <div className="bg-panel border border-accent rounded p-3 text-xs text-muted">
        Requires auto-labeling to have been run first to generate confidence scores.
      </div>

      <div className="bg-panel border border-accent rounded p-4 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Uncertainty method
          <select
            value={cfg.method}
            onChange={(e) => setCfg((c) => ({ ...c, method: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {['entropy','margin','least_confidence'].map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Frames to select
          <input type="number" min={1} max={10000}
            value={cfg.n_frames}
            onChange={(e) => setCfg((c) => ({ ...c, n_frames: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-24 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Uncertainty weight
          <input type="number" min={0} max={1} step={0.1}
            value={cfg.uncertainty_weight}
            onChange={(e) => setCfg((c) => ({ ...c, uncertainty_weight: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Diversity weight
          <input type="number" min={0} max={1} step={0.1}
            value={cfg.diversity_weight}
            onChange={(e) => setCfg((c) => ({ ...c, diversity_weight: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Scoring frames…' : 'Score Frames'}
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

      {result && (
        <div className="bg-panel border border-accent rounded p-4">
          <p className="text-gray-400 font-bold text-sm mb-3">Selection Results</p>
          <div className="grid grid-cols-2 gap-2 text-xs mb-4">
            <div><span className="text-muted">Total frames: </span><span>{result.total_frames}</span></div>
            <div><span className="text-muted">Selected: </span><span className="text-green-400">{result.selected_count}</span></div>
          </div>
          {budget && (
            <div className="bg-bg border border-accent rounded p-3 text-xs">
              <p className="text-gray-300 font-bold mb-1">Annotation Budget</p>
              <div className="flex gap-6">
                <div><span className="text-muted">Frames: </span><span>{budget.selected_frames}</span></div>
                <div><span className="text-muted">Est. time: </span><span>~{budget.estimated_annotation_hours}h</span></div>
                <div><span className="text-muted">Est. cost: </span><span>~${budget.estimated_cost_usd}</span></div>
                <div><span className="text-muted">Coverage: </span><span>{budget.coverage_percent?.toFixed(1)}%</span></div>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
