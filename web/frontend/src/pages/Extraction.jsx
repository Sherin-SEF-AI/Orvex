import { useState } from 'react'
import { runExtraction, getExtractionStatus } from '../api/client'
import { connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  frame_fps: 5.0,
  frame_format: 'jpg',
  frame_quality: 95,
  output_format: 'euroc',
  sync_devices: true,
  imu_interpolation: true,
}

export default function Extraction({ sessionId }) {
  const [cfg, setCfg]           = useState(DEFAULT_CFG)
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [status, setStatus]     = useState(null)
  const [stats, setStats]       = useState(null)   // { frames, imu_samples, gps_samples, elapsed_s, eta_s, output_size_mb }
  const [error, setError]       = useState(null)

  if (!sessionId) return <p className="text-muted text-sm">Select a session.</p>

  async function handleRun() {
    setRunning(true); setError(null); setLog([]); setProgress(0); setStats(null)
    try {
      const { task_id } = await runExtraction(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.elapsed_s != null || msg.eta_s != null || msg.frames != null) {
          setStats((prev) => ({ ...prev, ...msg }))
        }
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          getExtractionStatus(sessionId).then((d) => setStatus(d.status)).catch(() => {})
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  function fmtTime(s) {
    if (s == null) return '—'
    if (s < 60) return `${Math.round(s)}s`
    return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`
  }

  function field(key, label, type = 'text', opts = {}) {
    return (
      <label key={key} className="flex flex-col gap-1 text-xs text-muted">
        {label}
        <input
          type={type}
          value={cfg[key]}
          onChange={(e) => setCfg((c) => ({
            ...c,
            [key]: type === 'number' ? Number(e.target.value) : e.target.value,
          }))}
          {...opts}
          className="bg-panel border border-accent rounded px-2 py-1 text-sm text-gray-300
                     focus:outline-none focus:border-highlight w-32"
        />
      </label>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Extraction</h2>

      <div className="flex flex-wrap gap-4 bg-panel border border-accent rounded p-4">
        {field('frame_fps',    'Frame FPS',     'number', { min: 0.1, max: 60,  step: 0.5 })}
        {field('frame_quality','JPEG Quality',  'number', { min: 1,   max: 100, step: 1   })}
        <label className="flex flex-col gap-1 text-xs text-muted">
          Output Format
          <select
            value={cfg.output_format}
            onChange={(e) => setCfg((c) => ({ ...c, output_format: e.target.value }))}
            className="bg-panel border border-accent rounded px-2 py-1 text-sm text-gray-300
                       focus:outline-none focus:border-highlight"
          >
            <option value="euroc">EuRoC</option>
            <option value="custom">Custom</option>
          </select>
        </label>
        {[
          ['sync_devices',     'Sync devices'],
          ['imu_interpolation','IMU interpolation'],
        ].map(([k, label]) => (
          <label key={k} className="flex items-center gap-2 text-xs text-muted cursor-pointer">
            <input
              type="checkbox"
              checked={cfg[k]}
              onChange={(e) => setCfg((c) => ({ ...c, [k]: e.target.checked }))}
              className="accent-highlight"
            />
            {label}
          </label>
        ))}
      </div>

      <div className="flex items-center gap-3">
        {status && <span className="text-xs text-muted">Status: <b className="text-gray-300">{status}</b></span>}
        {error && <span className="text-red-400 text-xs">{error}</span>}
        <button onClick={handleRun} disabled={running}
          className="ml-auto text-sm px-4 py-2 bg-accent rounded hover:bg-highlight
                     disabled:opacity-50 font-semibold transition-colors">
          {running ? 'Extracting…' : 'Run Extraction'}
        </button>
      </div>

      {/* Stat cards — 2×3 grid */}
      {(running || stats) && (
        <div className="grid grid-cols-3 gap-3">
          {[
            { label: 'Frames',       value: stats?.frames        ?? '—' },
            { label: 'IMU samples',  value: stats?.imu_samples   ?? '—' },
            { label: 'GPS samples',  value: stats?.gps_samples   ?? '—' },
            { label: 'Elapsed',      value: fmtTime(stats?.elapsed_s) },
            { label: 'ETA',          value: fmtTime(stats?.eta_s) },
            { label: 'Output size',  value: stats?.output_size_mb != null ? `${stats.output_size_mb.toFixed(1)} MB` : '—' },
          ].map(({ label, value }) => (
            <div key={label} className="bg-panel border border-accent rounded p-3">
              <p className="text-xs text-muted mb-1">{label}</p>
              <p className="text-lg font-bold text-gray-200">{value}</p>
            </div>
          ))}
        </div>
      )}

      {(running || log.length > 0) && <ProgressLog lines={log} progress={progress} />}
    </div>
  )
}
