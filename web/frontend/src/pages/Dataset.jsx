import { useState, useEffect } from 'react'
import { getSessions, buildDataset, getProfiles, saveProfile } from '../api/client'
import { connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'
import StatusBadge from '../components/StatusBadge'

const STATUS_COLORS = {
  done:    '#22c55e',
  running: '#3b82f6',
  failed:  '#ef4444',
  pending: '#6b7280',
}

function SessionTimeline({ sessions }) {
  if (!sessions.length) return null
  const timestamps = sessions.map((s) => new Date(s.created_at).getTime())
  const minT = Math.min(...timestamps)
  const maxT = Math.max(...timestamps) + 1000   // avoid zero-width range

  return (
    <div className="bg-panel border border-accent rounded p-4">
      <p className="text-xs text-muted font-semibold mb-3 uppercase tracking-wide">Session Timeline</p>
      <div className="flex flex-col gap-1.5">
        {sessions.map((s) => {
          const t = new Date(s.created_at).getTime()
          const dur = (s.duration_seconds ?? 60) * 1000
          const left  = ((t - minT) / (maxT - minT + dur)) * 100
          const width = Math.max(1, (dur / (maxT - minT + dur)) * 100)
          const color = STATUS_COLORS[s.extraction_status] ?? STATUS_COLORS.pending
          return (
            <div key={s.id} className="flex items-center gap-2">
              <span className="text-xs text-muted w-28 truncate shrink-0">{s.name}</span>
              <div className="flex-1 relative h-5 bg-[#0f3460]/40 rounded overflow-hidden">
                <div
                  title={`${s.name} — ${s.extraction_status}`}
                  style={{ left: `${left}%`, width: `${width}%`, background: color }}
                  className="absolute top-0 bottom-0 rounded opacity-80 hover:opacity-100 transition-opacity"
                />
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

export default function Dataset() {
  const [sessions, setSessions]     = useState([])
  const [selected, setSelected]     = useState([])
  const [outputDir, setOutputDir]   = useState('')
  const [format, setFormat]         = useState('euroc')
  const [profiles, setProfiles]     = useState([])
  const [selProfile, setSelProfile] = useState('')
  const [log, setLog]               = useState([])
  const [progress, setProgress]     = useState(null)
  const [running, setRunning]       = useState(false)
  const [result, setResult]         = useState(null)
  const [integrity, setIntegrity]   = useState(null)
  const [error, setError]           = useState(null)

  useEffect(() => {
    getSessions().then(setSessions).catch(() => {})
    getProfiles().then(setProfiles).catch(() => {})
  }, [])

  function toggleSelect(id) {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    )
  }

  function applyProfile(name) {
    const p = profiles.find((x) => x.name === name)
    if (!p) return
    setFormat(p.output_format ?? format)
    setSelProfile(name)
  }

  async function handleSaveProfile() {
    const name = window.prompt('Profile name:')
    if (!name) return
    try {
      await saveProfile({ name, output_format: format, frame_fps: 5.0, frame_quality: 95,
                          blur_threshold: 100.0, dedup_threshold: 0.98 })
      const updated = await getProfiles()
      setProfiles(updated)
    } catch (e) { setError(e.message) }
  }

  async function handleBuild() {
    if (!selected.length) return
    if (!outputDir) { setError('Enter an output directory.'); return }
    setRunning(true); setError(null); setLog([]); setProgress(0); setResult(null); setIntegrity(null)

    try {
      const { task_id } = await buildDataset({
        session_ids: selected,
        output_format: format,
        output_dir: outputDir,
      })
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done') {
          setRunning(false); ws.close()
          if (msg.integrity) setIntegrity(msg.integrity)
        }
        if (msg.status === 'failed') { setRunning(false); ws.close() }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  return (
    <div className="flex flex-col gap-4 max-w-3xl">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Dataset Builder</h2>

      {/* Timeline */}
      <SessionTimeline sessions={sessions} />

      {/* Session selector */}
      <div className="bg-panel border border-accent rounded p-4">
        <p className="text-xs text-muted mb-2 font-semibold">Select sessions to include:</p>
        <div className="flex flex-col gap-2 max-h-48 overflow-y-auto">
          {sessions.length === 0 && <p className="text-muted text-xs">No sessions found.</p>}
          {sessions.map((s) => (
            <label key={s.id} className="flex items-center gap-3 cursor-pointer">
              <input type="checkbox" checked={selected.includes(s.id)}
                onChange={() => toggleSelect(s.id)} className="accent-highlight" />
              <span className="text-sm">{s.name}</span>
              <span className="text-xs text-muted">{s.environment} · {s.location}</span>
              <StatusBadge status={s.extraction_status} />
            </label>
          ))}
        </div>
      </div>

      {/* Config row */}
      <div className="flex flex-wrap gap-4 items-end">
        {profiles.length > 0 && (
          <label className="flex flex-col gap-1 text-xs text-muted">
            Profile
            <div className="flex gap-1">
              <select value={selProfile} onChange={(e) => applyProfile(e.target.value)}
                className="bg-panel border border-accent rounded px-2 py-1.5 text-sm text-gray-300
                           focus:outline-none focus:border-highlight">
                <option value="">— select —</option>
                {profiles.map((p) => (
                  <option key={p.name} value={p.name}>{p.name}</option>
                ))}
              </select>
              <button onClick={handleSaveProfile}
                className="text-xs px-2 py-1 border border-accent rounded hover:border-highlight transition-colors">
                Save…
              </button>
            </div>
          </label>
        )}
        <label className="flex flex-col gap-1 text-xs text-muted">
          Format
          <select value={format} onChange={(e) => setFormat(e.target.value)}
            className="bg-panel border border-accent rounded px-2 py-1.5 text-sm text-gray-300
                       focus:outline-none focus:border-highlight">
            <option value="euroc">EuRoC</option>
            <option value="rosbag2">ROS bag 2</option>
            <option value="hdf5">HDF5</option>
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted flex-1">
          Output directory (server-side path)
          <input value={outputDir} onChange={(e) => setOutputDir(e.target.value)}
            placeholder="/data/exports/my_dataset"
            className="bg-panel border border-accent rounded px-3 py-1.5 text-sm font-mono
                       text-gray-300 focus:outline-none focus:border-highlight" />
        </label>
        <button onClick={handleBuild} disabled={running || !selected.length}
          className="px-5 py-2 bg-green-700 hover:bg-green-600 text-white rounded font-semibold
                     text-sm disabled:opacity-50 transition-colors">
          {running ? 'Building…' : 'Build Dataset'}
        </button>
      </div>

      {error && <p className="text-red-400 text-xs">{error}</p>}
      {(running || log.length > 0) && <ProgressLog lines={log} progress={progress} />}

      {/* Integrity result */}
      {integrity && (
        <div className={`rounded border p-3 text-xs ${integrity.ok
          ? 'bg-green-950/40 border-green-700 text-green-300'
          : 'bg-yellow-950/40 border-yellow-700 text-yellow-300'}`}>
          <span className="font-semibold mr-2">Integrity: {integrity.ok ? 'OK' : 'WARN'}</span>
          {integrity.frame_count != null && (
            <span className="mr-3">{integrity.frame_count} frames · {integrity.imu_count} IMU rows</span>
          )}
          {integrity.gaps?.length > 0 && (
            <span className="text-yellow-400">{integrity.gaps.length} timestamp gap(s) detected</span>
          )}
          {integrity.warnings?.map((w, i) => (
            <p key={i} className="mt-1 text-yellow-400">{w}</p>
          ))}
        </div>
      )}
    </div>
  )
}
