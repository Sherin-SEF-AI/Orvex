import { useState, useEffect } from 'react'
import { runSlam, getSlamResults, getSlamInstallation, generateSlamConfig, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const MODES = ['mono','mono_inertial','stereo']

export default function SlamValidation({ sessionId }) {
  const [vocabulary, setVocabulary] = useState('')
  const [configYaml, setConfigYaml]  = useState('')
  const [mode, setMode]              = useState('mono_inertial')
  const [log, setLog]                = useState([])
  const [progress, setProgress]      = useState(null)
  const [running, setRunning]        = useState(false)
  const [result, setResult]          = useState(null)
  const [installation, setInstallation] = useState(null)
  const [error, setError]            = useState(null)

  useEffect(() => {
    getSlamInstallation().then(setInstallation).catch(() => {})
  }, [])

  if (!sessionId) return <p className="text-muted text-sm">Select a session first.</p>

  async function handleGenerateConfig() {
    try {
      const { config_path } = await generateSlamConfig(sessionId)
      setConfigYaml(config_path)
    } catch (e) { setError(e.message) }
  }

  async function handleRun() {
    if (!vocabulary || !configYaml) {
      setError('Vocabulary path and config YAML are required.')
      return
    }
    setRunning(true); setError(null); setLog([]); setProgress(0); setResult(null)
    try {
      const { task_id } = await runSlam(sessionId, {
        vocabulary_path: vocabulary,
        config_yaml_path: configYaml,
        mode,
      })
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getSlamResults(sessionId).then((d) => setResult(d.result)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  const metrics = result?.metrics

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">SLAM Validation</h2>

      {installation && (
        <div className="bg-panel border border-accent rounded p-3 text-xs">
          <p className="text-gray-400 mb-1">Installation status</p>
          <div className="flex gap-4">
            {Object.entries(installation).map(([k, v]) => (
              <span key={k}>
                <span className="text-muted">{k}: </span>
                <span className={v ? 'text-green-400' : 'text-red-400'}>{v ? '✓' : '✗'}</span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="bg-panel border border-accent rounded p-4 flex flex-col gap-3">
        <p className="text-xs text-muted">
          Requires EuRoC-format extraction (cam0/data/ + imu0/data.csv).
        </p>
        <label className="flex flex-col gap-1 text-xs text-muted">
          ORBSLAM3 vocabulary path
          <input
            value={vocabulary}
            onChange={(e) => setVocabulary(e.target.value)}
            placeholder="/path/to/ORBvoc.txt"
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          />
        </label>
        <div className="flex gap-2 items-end">
          <label className="flex flex-col gap-1 text-xs text-muted flex-1">
            Config YAML path
            <input
              value={configYaml}
              onChange={(e) => setConfigYaml(e.target.value)}
              placeholder="/path/to/config.yaml"
              className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
            />
          </label>
          <button
            onClick={handleGenerateConfig}
            className="px-3 py-1.5 bg-accent text-sm text-gray-300 rounded hover:bg-highlight/70"
          >
            Auto-generate from calibration
          </button>
        </div>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Mode
          <select
            value={mode}
            onChange={(e) => setMode(e.target.value)}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300 w-40"
          >
            {MODES.map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Running ORBSLAM3…' : 'Run SLAM Validation'}
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

      {result && metrics && (
        <div className="bg-panel border border-accent rounded p-4">
          <p className="text-gray-400 font-bold text-sm mb-3">Trajectory Metrics</p>
          <table className="text-xs w-full">
            <tbody>
              {[
                ['Distance', `${metrics.total_distance_m?.toFixed(2)} m`],
                ['Duration', `${metrics.duration_seconds?.toFixed(1)} s`],
                ['Avg speed', `${metrics.avg_speed_mps?.toFixed(2)} m/s`],
                ['Keyframes', metrics.keyframe_count],
                ['Map points', metrics.map_point_count],
                ['Tracking lost', metrics.tracking_lost_count],
                ['ATE RMSE', metrics.ate_rmse != null ? `${metrics.ate_rmse?.toFixed(4)} m` : '—'],
                ['Loop drift',
                  metrics.loop_closure_drift_m != null
                    ? `${metrics.loop_closure_drift_m?.toFixed(3)} m (${metrics.loop_closure_drift_percent?.toFixed(1)}%)`
                    : '—'
                ],
              ].map(([label, val]) => (
                <tr key={label} className="border-t border-accent/30">
                  <td className="py-1 text-muted pr-4">{label}</td>
                  <td className="py-1 text-gray-200">{val}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
