import { useState } from 'react'
import { runAudit, getAuditResults, addFile } from '../api/client'
import { connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'
import FileDropzone from '../components/FileDropzone'

export default function Audit({ sessionId }) {
  const [results, setResults]         = useState([])
  const [log, setLog]                 = useState([])
  const [progress, setProgress]       = useState(null)
  const [running, setRunning]         = useState(false)
  const [expandedWarnings, setExpWarn] = useState({})
  const [error, setError]             = useState(null)

  function exportJson() {
    const blob = new Blob([JSON.stringify(results, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `audit_${sessionId}.json`
    a.click()
    URL.revokeObjectURL(url)
  }

  function toggleWarnings(i) {
    setExpWarn((prev) => ({ ...prev, [i]: !prev[i] }))
  }

  if (!sessionId) return (
    <p className="text-muted text-sm">Select a session from the sidebar.</p>
  )

  async function handleAddFiles(paths) {
    for (const p of paths) {
      try { await addFile(sessionId, p) }
      catch (e) { setError(`Failed to add ${p}: ${e.message}`) }
    }
  }

  async function handleRun() {
    setRunning(true)
    setError(null)
    setLog([])
    setProgress(0)
    try {
      const { task_id } = await runAudit(sessionId)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false)
          ws.close()
          if (msg.status === 'done') {
            getAuditResults(sessionId).then(setResults).catch(() => {})
          }
        }
      })
    } catch (e) {
      setError(e.message)
      setRunning(false)
    }
  }

  async function handleLoadResults() {
    try { setResults(await getAuditResults(sessionId)) }
    catch (e) { setError(e.message) }
  }

  return (
    <div className="flex flex-col gap-4">
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide flex-1">
          File Audit
        </h2>
        <button onClick={handleLoadResults}
          className="text-xs px-3 py-1 border border-accent rounded hover:border-highlight">
          Refresh Results
        </button>
        {results.length > 0 && (
          <button onClick={exportJson}
            className="text-xs px-3 py-1 border border-accent rounded hover:border-highlight">
            Export JSON
          </button>
        )}
        <button onClick={handleRun} disabled={running}
          className="text-xs px-3 py-1 bg-accent rounded hover:bg-highlight
                     disabled:opacity-50 transition-colors font-semibold">
          {running ? 'Running…' : 'Run Audit'}
        </button>
      </div>

      <FileDropzone onPaths={handleAddFiles} />

      {error && <p className="text-red-400 text-xs">{error}</p>}
      {(running || log.length > 0) && <ProgressLog lines={log} progress={progress} />}

      {results.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse">
            <thead>
              <tr className="text-muted border-b border-accent">
                {['File','Device','Duration','IMU Hz','GPS Hz','FPS','Resolution','Size','Issues','Warnings'].map((h) => (
                  <th key={h} className="text-left px-2 py-1 font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {results.map((r, i) => {
                const hasIssues = r.issues?.length > 0
                const hasWarnings = r.warnings?.length > 0
                const rowBg = hasIssues
                  ? 'bg-red-950/40'
                  : r.chapter_files?.length > 0
                    ? 'bg-teal-950/40'
                    : ''
                return (
                  <tr key={i} className={`border-b border-accent/30 hover:brightness-110 ${rowBg}`}>
                    <td className="px-2 py-1 font-mono truncate max-w-xs">{r.file_path.split('/').pop()}</td>
                    <td className="px-2 py-1">{r.device_type}</td>
                    <td className="px-2 py-1">{r.duration_seconds?.toFixed(1)}s</td>
                    <td className={`px-2 py-1 ${r.has_imu ? 'text-green-400' : 'text-red-400'}`}>
                      {r.has_imu ? r.imu_rate_hz?.toFixed(0) : '✗'}
                    </td>
                    <td className={`px-2 py-1 ${r.has_gps ? 'text-green-400' : 'text-muted'}`}>
                      {r.has_gps ? r.gps_rate_hz?.toFixed(1) : '—'}
                    </td>
                    <td className="px-2 py-1">{r.video_fps?.toFixed(2)}</td>
                    <td className="px-2 py-1">{r.video_resolution?.[0]}×{r.video_resolution?.[1]}</td>
                    <td className="px-2 py-1">{r.file_size_mb?.toFixed(1)} MB</td>
                    <td className={`px-2 py-1 ${hasIssues ? 'text-yellow-400' : 'text-green-400'}`}>
                      {hasIssues ? r.issues.join('; ') : '—'}
                    </td>
                    <td className="px-2 py-1">
                      {hasWarnings ? (
                        <button
                          onClick={() => toggleWarnings(i)}
                          className="text-amber-400 hover:text-amber-200 underline cursor-pointer"
                        >
                          {r.warnings.length} note{r.warnings.length > 1 ? 's' : ''}
                          {expandedWarnings[i] ? ' ▲' : ' ▼'}
                        </button>
                      ) : (
                        <span className="text-muted">—</span>
                      )}
                      {expandedWarnings[i] && hasWarnings && (
                        <ul className="mt-1 text-amber-300 space-y-0.5 list-disc list-inside">
                          {r.warnings.map((w, j) => <li key={j}>{w}</li>)}
                        </ul>
                      )}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
