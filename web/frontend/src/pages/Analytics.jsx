import { useState } from 'react'
import { runAnalytics, getAnalyticsResults, generateReport, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORS = ['#e94560','#4fc3f7','#66bb6a','#ffa726','#ab47bc','#ef5350']

export default function Analytics({ sessionId }) {
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [result, setResult]     = useState(null)
  const [reportPath, setReportPath] = useState(null)
  const [error, setError]       = useState(null)

  if (!sessionId) return <p className="text-muted text-sm">Select a session first.</p>

  async function handleRun() {
    setRunning(true); setError(null); setLog([]); setProgress(0); setResult(null)
    try {
      const { task_id } = await runAnalytics(sessionId, { sample_n: 500 })
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getAnalyticsResults(sessionId).then((d) => setResult(d.result)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  async function handleReport() {
    try {
      const { report_path } = await generateReport(sessionId)
      setReportPath(report_path)
    } catch (e) { setError(e.message) }
  }

  const diversity = result?.diversity
  const coverage  = result?.coverage

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Analytics</h2>

      <div className="flex gap-2">
        <button
          onClick={handleRun}
          disabled={running}
          className="px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                     disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
        >
          {running ? 'Computing…' : 'Compute Analytics'}
        </button>
        {result && (
          <button
            onClick={handleReport}
            className="px-4 py-2 bg-accent text-gray-300 rounded text-sm hover:bg-highlight/70"
          >
            Generate HTML Report
          </button>
        )}
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}
      {reportPath && <p className="text-green-400 text-xs">Report saved: {reportPath}</p>}

      {progress !== null && (
        <div className="bg-panel border border-accent rounded p-1">
          <div className="h-2 bg-bg rounded overflow-hidden">
            <div className="h-2 bg-highlight rounded transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      <ProgressLog lines={log} />

      {diversity && (
        <div className="bg-panel border border-accent rounded p-4">
          <p className="text-gray-400 font-bold text-sm mb-3">Scene Diversity</p>
          <div className="grid grid-cols-2 gap-6">
            {/* Lighting distribution */}
            <div>
              <p className="text-xs text-muted mb-2">Lighting distribution</p>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={Object.entries(diversity.lighting_distribution || {}).map(([k, v]) => ({ name: k, value: Number(v.toFixed(1)) }))}>
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#666680' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#666680' }} unit="%" />
                  <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #0f3460' }} />
                  <Bar dataKey="value">
                    {['bright','normal','dark'].map((_, i) => (
                      <Cell key={i} fill={COLORS[i]} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
            {/* Scene types */}
            <div>
              <p className="text-xs text-muted mb-2">Estimated scene types</p>
              <ResponsiveContainer width="100%" height={120}>
                <BarChart data={Object.entries(diversity.estimated_scene_types || {}).map(([k, v]) => ({ name: k, value: Number(v.toFixed(1)) }))}>
                  <XAxis dataKey="name" tick={{ fontSize: 10, fill: '#666680' }} />
                  <YAxis tick={{ fontSize: 10, fill: '#666680' }} unit="%" />
                  <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #0f3460' }} />
                  <Bar dataKey="value" fill="#4fc3f7" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="mt-3 grid grid-cols-4 gap-2 text-xs">
            {Object.entries(diversity.brightness_stats || {}).map(([k, v]) => (
              <div key={k}>
                <span className="text-muted">{k}: </span>
                <span>{typeof v === 'number' ? v.toFixed(1) : v}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {coverage && (
        <div className="bg-panel border border-accent rounded p-4">
          <p className="text-gray-400 font-bold text-sm mb-3">Geographic Coverage</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            {[
              ['Total distance', `${coverage.total_distance_km?.toFixed(2)} km`],
              ['Unique grid cells (10m)', coverage.unique_grid_cells],
              ['Avg speed', `${coverage.avg_speed_mps?.toFixed(1)} m/s`],
              ['Stationary time', `${coverage.stationary_time_percent?.toFixed(1)}%`],
            ].map(([label, val]) => (
              <div key={label}>
                <span className="text-muted">{label}: </span>
                <span>{val}</span>
              </div>
            ))}
          </div>
          {coverage.coverage_map_path && (
            <p className="text-xs text-muted mt-2">
              Coverage map: <span className="text-gray-300">{coverage.coverage_map_path}</span>
            </p>
          )}
        </div>
      )}
    </div>
  )
}
