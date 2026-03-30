import { useState, useEffect } from 'react'
import { getExperimentsHealth, listRuns, getRun, compareRuns, launchMlflowUI } from '../api/client'

function MapBadge({ value }) {
  if (value === undefined || value === null) return <span className="text-muted">—</span>
  const n = parseFloat(value)
  const color = n < 0.3 ? 'text-red-400' : n < 0.5 ? 'text-yellow-400' : 'text-green-400'
  return <span className={color}>{n.toFixed(3)}</span>
}

export default function Experiments() {
  const [health, setHealth]         = useState(null)
  const [runs, setRuns]             = useState([])
  const [selected, setSelected]     = useState(new Set())
  const [detail, setDetail]         = useState(null)
  const [comparison, setComparison] = useState(null)
  const [error, setError]           = useState(null)
  const [status, setStatus]         = useState('')

  useEffect(() => {
    getExperimentsHealth().then(setHealth).catch(() => {})
    loadRuns()
  }, [])

  async function loadRuns() {
    try {
      const r = await listRuns()
      setRuns(r)
    } catch (e) { setError(e.message) }
  }

  function toggleSelect(runId) {
    setSelected((s) => {
      const ns = new Set(s)
      ns.has(runId) ? ns.delete(runId) : ns.add(runId)
      return ns
    })
    setComparison(null)
    if (selected.size === 0) {
      getRun(runId).then(setDetail).catch(() => {})
    }
  }

  async function handleCompare() {
    const ids = [...selected]
    if (ids.length < 2) return
    try {
      const c = await compareRuns(ids.join(','))
      setComparison(c)
    } catch (e) { setError(e.message) }
  }

  async function handleLaunchUI() {
    try {
      const r = await launchMlflowUI()
      setStatus(`MLflow UI launching at ${r.url}`)
      setTimeout(() => window.open(r.url, '_blank'), 2500)
    } catch (e) { setError(e.message) }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Experiments (MLflow)</h2>
        {health && (
          <span className={`px-2 py-0.5 rounded text-xs ${health.mlflow ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
            {health.mlflow ? '✓ MLflow Ready' : '⚠ Not installed'}
          </span>
        )}
      </div>

      <div className="flex gap-2">
        <button onClick={handleLaunchUI}
          className="px-3 py-1.5 bg-accent text-gray-300 rounded text-sm hover:bg-hover">
          Launch MLflow UI
        </button>
        <button onClick={loadRuns}
          className="px-3 py-1.5 bg-panel border border-accent text-gray-300 rounded text-sm hover:bg-hover">
          Refresh
        </button>
        {selected.size >= 2 && (
          <button onClick={handleCompare}
            className="px-3 py-1.5 bg-blue-800 text-blue-200 rounded text-sm hover:bg-blue-700">
            Compare {selected.size} runs
          </button>
        )}
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}
      {status && <p className="text-green-400 text-sm">{status}</p>}

      {!health?.mlflow && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 text-xs text-yellow-300">
          Install MLflow: <code>pip install mlflow</code>. Runs are tracked automatically during training.
        </div>
      )}

      {runs.length === 0 ? (
        <p className="text-muted text-sm">No runs yet. Training runs are logged automatically.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="border-b border-accent text-muted text-left">
                <th className="px-2 py-1 w-6"></th>
                <th className="px-2 py-1">Name</th>
                <th className="px-2 py-1">Status</th>
                <th className="px-2 py-1">mAP50</th>
                <th className="px-2 py-1">mAP50-95</th>
                <th className="px-2 py-1">Precision</th>
                <th className="px-2 py-1">Recall</th>
                <th className="px-2 py-1">Model</th>
                <th className="px-2 py-1">Date</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((r) => (
                <tr key={r.run_id}
                  onClick={() => toggleSelect(r.run_id)}
                  className={`cursor-pointer border-b border-accent/30 hover:bg-hover
                    ${selected.has(r.run_id) ? 'bg-accent/40' : ''}`}
                >
                  <td className="px-2 py-1">
                    <input type="checkbox" readOnly checked={selected.has(r.run_id)} className="pointer-events-none" />
                  </td>
                  <td className="px-2 py-1 text-gray-300 font-mono">{r.run_name}</td>
                  <td className="px-2 py-1">
                    <span className={`px-1.5 rounded ${r.status === 'FINISHED' ? 'bg-green-900 text-green-300' : 'bg-gray-800 text-gray-400'}`}>
                      {r.status}
                    </span>
                  </td>
                  <td className="px-2 py-1"><MapBadge value={r.metrics?.['val/mAP50']} /></td>
                  <td className="px-2 py-1"><MapBadge value={r.metrics?.['val/mAP50-95']} /></td>
                  <td className="px-2 py-1 text-gray-300">{r.metrics?.['val/precision']?.toFixed(3) ?? '—'}</td>
                  <td className="px-2 py-1 text-gray-300">{r.metrics?.['val/recall']?.toFixed(3) ?? '—'}</td>
                  <td className="px-2 py-1 text-gray-400">{r.params?.model_variant ?? '—'}</td>
                  <td className="px-2 py-1 text-gray-500">{new Date(r.start_time).toLocaleDateString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {detail && selected.size === 1 && (
        <div className="bg-panel border border-accent rounded p-4 text-xs">
          <p className="font-bold text-gray-400 mb-2">Run: {detail.run_name}</p>
          <div className="grid grid-cols-2 gap-1 text-gray-300">
            {Object.entries(detail.params || {}).map(([k, v]) => (
              <div key={k}><span className="text-muted">{k}: </span>{v}</div>
            ))}
          </div>
        </div>
      )}

      {comparison && (
        <div className="bg-panel border border-accent rounded p-4 text-xs">
          <p className="font-bold text-gray-400 mb-2">
            Best run: <code className="text-green-400">{comparison.best_run_id?.slice(0, 8)}…</code>
          </p>
          <table className="w-full border-collapse">
            <thead>
              <tr className="border-b border-accent text-muted">
                <th className="px-2 py-1 text-left">Metric</th>
                {comparison.runs?.map((r) => (
                  <th key={r.run_id} className="px-2 py-1 text-left">{r.run_name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(comparison.metric_comparison || {}).map(([metric, vals]) => (
                <tr key={metric} className="border-b border-accent/20">
                  <td className="px-2 py-1 text-muted">{metric}</td>
                  {comparison.runs?.map((r) => (
                    <td key={r.run_id} className="px-2 py-1 text-gray-300">
                      {typeof vals?.[r.run_id] === 'number' ? vals[r.run_id].toFixed(4) : '—'}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
