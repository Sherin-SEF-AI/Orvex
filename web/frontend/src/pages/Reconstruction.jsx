import { useState, useEffect } from 'react'
import { runReconstruction, getReconstructionResults, getColmapInstallation, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  every_nth: 6,
  use_gpu: true,
  camera_model: 'OPENCV_FISHEYE',
  max_image_size: 1600,
}

const CAMERA_MODELS = ['OPENCV_FISHEYE', 'OPENCV', 'SIMPLE_RADIAL', 'RADIAL', 'PINHOLE']

export default function Reconstruction({ sessionId }) {
  const [cfg, setCfg]             = useState(DEFAULT_CFG)
  const [log, setLog]             = useState([])
  const [progress, setProgress]   = useState(null)
  const [running, setRunning]     = useState(false)
  const [result, setResult]       = useState(null)
  const [installation, setInstallation] = useState(null)
  const [error, setError]         = useState(null)

  useEffect(() => {
    getColmapInstallation().then(setInstallation).catch(() => {})
  }, [])

  if (!sessionId) return <p className="text-muted text-sm">Select a session first.</p>

  async function handleRun() {
    setRunning(true); setError(null); setLog([]); setProgress(0); setResult(null)
    try {
      const { task_id } = await runReconstruction(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getReconstructionResults(sessionId).then((d) => setResult(d.result)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  const matcherLabel = cfg.every_nth <= 1 ? 'exhaustive (< 100 frames)' : 'sequential'
  const coverage = result
    ? ((result.num_images_registered / Math.max(result.num_images_total, 1)) * 100).toFixed(1)
    : null

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">3D Reconstruction</h2>

      {installation && (
        <div className="bg-panel border border-accent rounded p-3 text-xs">
          <span className="text-muted">COLMAP: </span>
          <span className={installation.colmap ? 'text-green-400' : 'text-red-400'}>
            {installation.colmap ? '✓ installed' : '✗ not found'}
          </span>
        </div>
      )}

      <div className="bg-panel border border-accent rounded p-4 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-muted">
          Every Nth frame
          <input type="number" min={1} max={30}
            value={cfg.every_nth}
            onChange={(e) => setCfg((c) => ({ ...c, every_nth: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Camera model
          <select
            value={cfg.camera_model}
            onChange={(e) => setCfg((c) => ({ ...c, camera_model: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {CAMERA_MODELS.map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Max image size (px)
          <input type="number" min={640} max={4096} step={64}
            value={cfg.max_image_size}
            onChange={(e) => setCfg((c) => ({ ...c, max_image_size: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm w-24 text-gray-300"
          />
        </label>
        <label className="flex items-center gap-2 text-xs text-muted mt-4">
          <input type="checkbox" checked={cfg.use_gpu}
            onChange={(e) => setCfg((c) => ({ ...c, use_gpu: e.target.checked }))}
          />
          Use GPU
        </label>
      </div>

      <p className="text-xs text-muted">Feature matcher: <span className="text-gray-300">{matcherLabel}</span></p>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Running COLMAP…' : 'Start Reconstruction'}
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
        <>
          {coverage < 80 && (
            <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 text-xs text-yellow-300">
              ⚠ Coverage {coverage}% — fewer than 80% of images were registered. Consider using a smaller
              every-nth value or a wider camera model.
            </div>
          )}
          <div className="bg-panel border border-accent rounded p-4">
            <p className="text-gray-400 font-bold text-sm mb-3">Reconstruction Results</p>
            <table className="text-xs w-full">
              <tbody>
                {[
                  ['Images registered', `${result.num_images_registered} / ${result.num_images_total}`],
                  ['Coverage', `${coverage}%`],
                  ['3D points', result.num_points3d?.toLocaleString()],
                  ['Reprojection error', `${result.mean_reprojection_error?.toFixed(3)} px`],
                  ['PLY export', result.ply_path || '—'],
                ].map(([label, val]) => (
                  <tr key={label} className="border-t border-accent/30">
                    <td className="py-1 text-muted pr-4">{label}</td>
                    <td className="py-1 text-gray-200">{val}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
