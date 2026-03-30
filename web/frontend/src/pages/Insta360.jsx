import { useState, useEffect, useRef } from 'react'
import {
  scanINSV, validateINSV, processINSV, getINSVTask,
  getINSVGPS, getINSVIMU, getINSVFrames, getINSVManifest,
  getINSVHealth, connectTaskWS,
} from '../api/client'
import ProgressLog from '../components/ProgressLog'

// ---------------------------------------------------------------------------
// Default processing config
// ---------------------------------------------------------------------------

const DEFAULT_CFG = {
  output_width: 7680,
  output_height: 3840,
  stitch_crf: 10,
  fisheye_fov: 210.0,
  perspective_width: 2160,
  perspective_height: 2160,
  h_fov: 110.0,
  v_fov: 110.0,
  perspective_crf: 15,
  views: ['front', 'right', 'rear', 'left'],
  use_gpu: false,
  frame_fps: 5.0,
  frame_format: 'jpg',
  frame_quality: 95,
  pitch_correction_deg: 0.0,
  roll_correction_deg: 0.0,
  keep_equirect_video: true,
  keep_perspective_videos: false,
  output_format: 'euroc',
}

const RESOLUTION_PRESETS = [
  { label: '8K (7680×3840)', w: 7680, h: 3840 },
  { label: '4K (3840×1920)', w: 3840, h: 1920 },
  { label: '2K (1920×960)',  w: 1920, h: 960 },
]

const STAGES = ['Telemetry', 'Stitching', 'Perspective', 'Frames', 'Dataset']

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function INSVPairList({ pairs, selected, onSelect }) {
  if (!pairs.length) return (
    <p className="text-xs text-gray-500 italic">No INSV pairs found. Enter a folder path and scan.</p>
  )
  return (
    <ul className="flex flex-col gap-1">
      {pairs.map((p) => (
        <li
          key={p.base_name}
          onClick={() => onSelect(p)}
          className={`cursor-pointer rounded px-2 py-1.5 text-xs flex items-center gap-2 border transition-colors ${
            selected?.base_name === p.base_name
              ? 'bg-accent/20 border-accent text-gray-200'
              : 'bg-panel border-transparent text-gray-400 hover:border-accent/50'
          }`}
        >
          <span className={p.issues.length ? 'text-yellow-400' : 'text-green-400'}>
            {p.issues.length ? '⚠' : '✓'}
          </span>
          <span className="font-mono flex-1 truncate">{p.base_name}</span>
          <span className="text-gray-500">{p.duration_seconds.toFixed(1)}s</span>
          {p.has_gps && <span className="text-blue-400 text-[10px]">GPS</span>}
          {p.has_imu && <span className="text-purple-400 text-[10px]">IMU</span>}
        </li>
      ))}
    </ul>
  )
}

function PipelineStages({ stage, progress }) {
  return (
    <div className="flex flex-col gap-1.5">
      {STAGES.map((s) => {
        const active = stage && stage.toLowerCase().includes(s.toLowerCase())
        const done = progress >= (STAGES.indexOf(s) + 1) * 20
        return (
          <div key={s} className={`flex items-center gap-2 text-xs px-2 py-1 rounded ${
            active ? 'bg-accent/20 text-accent' : done ? 'text-green-400' : 'text-gray-500'
          }`}>
            <span>{done && !active ? '✓' : active ? '⟳' : '○'}</span>
            <span>{s}</span>
          </div>
        )
      })}
    </div>
  )
}

function FourViewGrid({ sessionId, frameIndex }) {
  const VIEWS = ['front', 'right', 'rear', 'left']
  const [frameNames, setFrameNames] = useState({ front: [], right: [], rear: [], left: [] })

  useEffect(() => {
    if (!sessionId) return
    Promise.all(
      VIEWS.map((v) =>
        getINSVFrames(sessionId, v, 1).then((d) => [v, d.frames]).catch(() => [v, []])
      )
    ).then((pairs) => {
      const m = {}
      pairs.forEach(([v, frames]) => { m[v] = frames })
      setFrameNames(m)
    })
  }, [sessionId])

  if (!sessionId) return null

  const baseUrl = `/insta360/${sessionId}/frames`
  return (
    <div className="grid grid-cols-2 gap-1">
      {VIEWS.map((v) => {
        const name = frameNames[v]?.[frameIndex] || null
        return (
          <div key={v} className="relative bg-black rounded overflow-hidden aspect-square">
            {name ? (
              <img
                src={`${baseUrl}/${v}/${name}`}
                alt={v}
                className="w-full h-full object-cover"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center text-gray-600 text-xs">
                {v.toUpperCase()}
              </div>
            )}
            <span className="absolute bottom-1 left-1 text-[10px] text-white/60 font-mono uppercase">
              {v}
            </span>
          </div>
        )
      })}
    </div>
  )
}

function TelemetryPanel({ sessionId }) {
  const [gps, setGps]   = useState([])
  const [imu, setImu]   = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    if (!sessionId) return
    setLoading(true)
    Promise.all([
      getINSVGPS(sessionId).catch(() => []),
      getINSVIMU(sessionId, 1).catch(() => []),
    ]).then(([g, im]) => {
      setGps(g)
      setImu(im)
      setLoading(false)
    })
  }, [sessionId])

  if (!sessionId) return null
  if (loading) return <p className="text-xs text-gray-500">Loading telemetry…</p>

  return (
    <div className="flex flex-col gap-4">
      <div>
        <p className="text-xs text-gray-400 mb-1 font-medium">GPS — {gps.length} samples</p>
        {gps.length ? (
          <div className="overflow-auto max-h-48 rounded border border-accent/20">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-accent/20">
                  <th className="text-left px-2 py-1">Lat</th>
                  <th className="text-left px-2 py-1">Lon</th>
                  <th className="text-left px-2 py-1">Alt (m)</th>
                  <th className="text-left px-2 py-1">Speed (m/s)</th>
                </tr>
              </thead>
              <tbody>
                {gps.slice(0, 20).map((s, i) => (
                  <tr key={i} className="text-gray-300 border-b border-accent/10">
                    <td className="px-2 py-0.5">{s.latitude.toFixed(6)}</td>
                    <td className="px-2 py-0.5">{s.longitude.toFixed(6)}</td>
                    <td className="px-2 py-0.5">{s.altitude_m.toFixed(1)}</td>
                    <td className="px-2 py-0.5">{s.speed_mps.toFixed(2)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-yellow-600">
            No GPS data. GPS requires a GPS remote or Insta360 app with screen-on during recording.
          </p>
        )}
      </div>

      <div>
        <p className="text-xs text-gray-400 mb-1 font-medium">IMU — {imu.length} samples (page 1)</p>
        {imu.length ? (
          <div className="overflow-auto max-h-48 rounded border border-accent/20">
            <table className="w-full text-xs font-mono">
              <thead>
                <tr className="text-gray-500 border-b border-accent/20">
                  <th className="text-left px-2 py-1">ts (ns)</th>
                  <th className="text-left px-2 py-1">ax</th>
                  <th className="text-left px-2 py-1">ay</th>
                  <th className="text-left px-2 py-1">az</th>
                  <th className="text-left px-2 py-1">gx</th>
                  <th className="text-left px-2 py-1">gy</th>
                  <th className="text-left px-2 py-1">gz</th>
                </tr>
              </thead>
              <tbody>
                {imu.slice(0, 20).map((s, i) => (
                  <tr key={i} className="text-gray-300 border-b border-accent/10">
                    <td className="px-2 py-0.5">{s.timestamp_ns}</td>
                    <td className="px-2 py-0.5">{s.accel_x?.toFixed(3)}</td>
                    <td className="px-2 py-0.5">{s.accel_y?.toFixed(3)}</td>
                    <td className="px-2 py-0.5">{s.accel_z?.toFixed(3)}</td>
                    <td className="px-2 py-0.5">{s.gyro_x?.toFixed(4)}</td>
                    <td className="px-2 py-0.5">{s.gyro_y?.toFixed(4)}</td>
                    <td className="px-2 py-0.5">{s.gyro_z?.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p className="text-xs text-yellow-600">No IMU data extracted.</p>
        )}
      </div>
    </div>
  )
}

function ManifestPanel({ sessionId }) {
  const [manifest, setManifest] = useState(null)

  useEffect(() => {
    if (!sessionId) return
    getINSVManifest(sessionId).then(setManifest).catch(() => {})
  }, [sessionId])

  if (!manifest) return <p className="text-xs text-gray-500">No manifest. Run processing first.</p>

  return (
    <pre className="text-xs text-gray-300 font-mono overflow-auto max-h-96 bg-black/30 rounded p-3">
      {JSON.stringify(manifest, null, 2)}
    </pre>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function Insta360() {
  const [folderPath, setFolderPath]   = useState('')
  const [outputDir, setOutputDir]     = useState('')
  const [pairs, setPairs]             = useState([])
  const [selected, setSelected]       = useState(null)
  const [cfg, setCfg]                 = useState(DEFAULT_CFG)
  const [validationIssues, setValidationIssues] = useState([])
  const [taskId, setTaskId]           = useState(null)
  const [stage, setStage]             = useState('')
  const [progress, setProgress]       = useState(0)
  const [running, setRunning]         = useState(false)
  const [log, setLog]                 = useState([])
  const [error, setError]             = useState(null)
  const [resultSessionId, setResultSessionId] = useState(null)
  const [results, setResults]         = useState(null)
  const [activeTab, setActiveTab]     = useState('4view')
  const [frameIndex, setFrameIndex]   = useState(0)
  const [health, setHealth]           = useState(null)
  const [diskWarning, setDiskWarning] = useState(null)
  const wsRef = useRef(null)

  // Load health on mount
  useEffect(() => {
    getINSVHealth().then(setHealth).catch(() => {})
  }, [])

  // Disk usage estimate warning
  useEffect(() => {
    if (!selected) { setDiskWarning(null); return }
    const estimatedGB = (selected.file_size_mb / 1024) * 8
    if (estimatedGB > 50) {
      setDiskWarning(`Estimated disk usage ~${estimatedGB.toFixed(0)} GB. Ensure sufficient space.`)
    } else {
      setDiskWarning(null)
    }
  }, [selected])

  async function handleScan() {
    if (!folderPath.trim()) return
    setError(null); setPairs([]); setSelected(null)
    try {
      const data = await scanINSV({ folder_path: folderPath.trim() })
      setPairs(data)
      if (!data.length) setError('No INSV pairs found in the specified folder.')
    } catch (e) { setError(e.message) }
  }

  async function handleValidate() {
    if (!selected) return
    setValidationIssues([])
    try {
      const { valid, issues } = await validateINSV({
        pair: selected,
        output_dir: outputDir || folderPath,
      })
      setValidationIssues(issues)
      if (valid) setLog((l) => [...l, '✓ Pre-flight validation passed.'])
      else setLog((l) => [...l, `⚠ Validation issues: ${issues.join('; ')}`])
    } catch (e) { setError(e.message) }
  }

  async function handleProcess() {
    if (!selected) return
    const outDir = outputDir.trim() || folderPath.trim()
    const sid = `insta360_${selected.base_name}`
    setRunning(true); setError(null); setLog([]); setProgress(0); setStage(''); setResults(null)
    try {
      const { task_id } = await processINSV({
        pair: selected,
        config: cfg,
        output_dir: outDir,
        session_id: sid,
      })
      setTaskId(task_id)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) {
          setLog((l) => [...l, msg.message])
          setStage(msg.message)
        }
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false)
          ws.close()
          if (msg.status === 'done') {
            setResultSessionId(sid)
            getINSVTask(task_id)
              .then((d) => { if (d.result) setResults(d.result) })
              .catch(() => {})
          }
          if (msg.status === 'failed') setError(msg.message || 'Pipeline failed.')
        }
      })
      wsRef.current = ws
    } catch (e) { setError(e.message); setRunning(false) }
  }

  function handleCancel() {
    wsRef.current?.close()
    setRunning(false)
  }

  const resPreset = RESOLUTION_PRESETS.find(
    (p) => p.w === cfg.output_width && p.h === cfg.output_height
  )

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">
        Insta360 X4 — 360° Processing
      </h2>

      {/* Health banner */}
      {health && (
        <div className={`rounded p-2 text-xs border ${
          health.ffmpeg?.minimum_met && health.exiftool?.available
            ? 'bg-green-900/20 border-green-700 text-green-300'
            : 'bg-red-900/20 border-red-700 text-red-300'
        }`}>
          {health.ffmpeg?.minimum_met
            ? `ffmpeg ${health.ffmpeg.ffmpeg_version} ✓`
            : `ffmpeg missing or too old (need ≥4.3)`}
          {' · '}
          {health.exiftool?.available
            ? `exiftool ${health.exiftool.version} ✓`
            : 'exiftool not found'}
        </div>
      )}

      {diskWarning && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded p-2 text-xs text-yellow-300">
          ⚠ {diskWarning}
        </div>
      )}

      {error && (
        <div className="bg-red-900/30 border border-red-700 rounded p-2 text-xs text-red-300">
          {error}
        </div>
      )}

      {validationIssues.length > 0 && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded p-2 text-xs text-yellow-200">
          <strong>Validation issues:</strong>
          <ul className="list-disc ml-4 mt-1">
            {validationIssues.map((i, idx) => <li key={idx}>{i}</li>)}
          </ul>
        </div>
      )}

      {/* Source + pair list */}
      <div className="bg-panel border border-accent/30 rounded p-4 flex flex-col gap-3">
        <p className="text-xs text-gray-400 font-medium">Source Folder</p>
        <div className="flex gap-2">
          <input
            value={folderPath}
            onChange={(e) => setFolderPath(e.target.value)}
            placeholder="/path/to/INSV/files"
            className="flex-1 bg-bg border border-accent/40 rounded px-3 py-1.5 text-sm font-mono text-gray-200 placeholder-gray-600"
          />
          <button
            onClick={handleScan}
            className="px-3 py-1.5 bg-accent/20 border border-accent rounded text-xs hover:bg-accent/30 text-gray-300"
          >
            Scan
          </button>
        </div>
        <INSVPairList pairs={pairs} selected={selected} onSelect={setSelected} />
      </div>

      {/* Config */}
      <div className="bg-panel border border-accent/30 rounded p-4 flex flex-wrap gap-4">
        <label className="flex flex-col gap-1 text-xs text-gray-400">
          Equirect Resolution
          <select
            value={resPreset?.label || 'custom'}
            onChange={(e) => {
              const p = RESOLUTION_PRESETS.find((r) => r.label === e.target.value)
              if (p) setCfg((c) => ({ ...c, output_width: p.w, output_height: p.h }))
            }}
            className="bg-bg border border-accent/40 rounded px-2 py-1 text-sm text-gray-300"
          >
            {RESOLUTION_PRESETS.map((p) => <option key={p.label}>{p.label}</option>)}
          </select>
        </label>

        <label className="flex flex-col gap-1 text-xs text-gray-400">
          Fisheye FOV (°)
          <input type="number" step="0.5" min={180} max={220}
            value={cfg.fisheye_fov}
            onChange={(e) => setCfg((c) => ({ ...c, fisheye_fov: Number(e.target.value) }))}
            className="bg-bg border border-accent/40 rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-gray-400">
          Stitch CRF (0–30)
          <input type="number" min={0} max={30}
            value={cfg.stitch_crf}
            onChange={(e) => setCfg((c) => ({ ...c, stitch_crf: Number(e.target.value) }))}
            className="bg-bg border border-accent/40 rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-gray-400">
          Persp. CRF (0–30)
          <input type="number" min={0} max={30}
            value={cfg.perspective_crf}
            onChange={(e) => setCfg((c) => ({ ...c, perspective_crf: Number(e.target.value) }))}
            className="bg-bg border border-accent/40 rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-gray-400">
          H-FOV (°)
          <input type="number" step="1" min={60} max={180}
            value={cfg.h_fov}
            onChange={(e) => setCfg((c) => ({ ...c, h_fov: Number(e.target.value) }))}
            className="bg-bg border border-accent/40 rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-gray-400">
          V-FOV (°)
          <input type="number" step="1" min={60} max={180}
            value={cfg.v_fov}
            onChange={(e) => setCfg((c) => ({ ...c, v_fov: Number(e.target.value) }))}
            className="bg-bg border border-accent/40 rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-gray-400">
          Frame FPS
          <input type="number" step="0.5" min={0.5} max={30}
            value={cfg.frame_fps}
            onChange={(e) => setCfg((c) => ({ ...c, frame_fps: Number(e.target.value) }))}
            className="bg-bg border border-accent/40 rounded px-2 py-1 text-sm w-20 text-gray-300"
          />
        </label>

        <label className="flex flex-col gap-1 text-xs text-gray-400">
          Frame Format
          <select
            value={cfg.frame_format}
            onChange={(e) => setCfg((c) => ({ ...c, frame_format: e.target.value }))}
            className="bg-bg border border-accent/40 rounded px-2 py-1 text-sm text-gray-300"
          >
            {['jpg', 'png'].map((f) => <option key={f}>{f}</option>)}
          </select>
        </label>

        <label className="flex items-center gap-2 text-xs text-gray-400 mt-4 self-end">
          <input type="checkbox" checked={cfg.use_gpu}
            onChange={(e) => setCfg((c) => ({ ...c, use_gpu: e.target.checked }))}
          />
          Use GPU (h264_nvenc)
        </label>

        <label className="flex items-center gap-2 text-xs text-gray-400 mt-4 self-end">
          <input type="checkbox" checked={cfg.keep_equirect_video}
            onChange={(e) => setCfg((c) => ({ ...c, keep_equirect_video: e.target.checked }))}
          />
          Keep equirect video
        </label>

        <label className="flex items-center gap-2 text-xs text-gray-400 mt-4 self-end">
          <input type="checkbox" checked={cfg.keep_perspective_videos}
            onChange={(e) => setCfg((c) => ({ ...c, keep_perspective_videos: e.target.checked }))}
          />
          Keep perspective videos
        </label>
      </div>

      {/* Output dir */}
      <div className="bg-panel border border-accent/30 rounded p-4 flex gap-2 items-center">
        <label className="text-xs text-gray-400 whitespace-nowrap">Output directory</label>
        <input
          value={outputDir}
          onChange={(e) => setOutputDir(e.target.value)}
          placeholder="(defaults to source folder)"
          className="flex-1 bg-bg border border-accent/40 rounded px-3 py-1.5 text-sm font-mono text-gray-200 placeholder-gray-600"
        />
      </div>

      {/* Actions */}
      <div className="flex gap-2">
        <button
          onClick={handleValidate}
          disabled={!selected || running}
          className="px-4 py-2 bg-panel border border-accent/40 rounded text-xs text-gray-300 hover:bg-accent/10 disabled:opacity-40"
        >
          Validate
        </button>
        <button
          onClick={handleProcess}
          disabled={!selected || running}
          className="px-4 py-2 bg-accent/20 border border-accent rounded text-xs text-gray-200 hover:bg-accent/30 disabled:opacity-40"
        >
          {running ? 'Processing…' : 'Start Pipeline'}
        </button>
        {running && (
          <button
            onClick={handleCancel}
            className="px-4 py-2 bg-red-900/20 border border-red-700 rounded text-xs text-red-300 hover:bg-red-900/40"
          >
            Cancel
          </button>
        )}
      </div>

      {/* Progress */}
      {(running || progress > 0) && (
        <div className="bg-panel border border-accent/30 rounded p-4 flex flex-col gap-3">
          <div className="flex items-center justify-between text-xs text-gray-400">
            <span>{stage || 'Running…'}</span>
            <span>{progress}%</span>
          </div>
          <div className="h-2 bg-black/40 rounded overflow-hidden">
            <div
              className="h-full bg-accent transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
          <PipelineStages stage={stage} progress={progress} />
        </div>
      )}

      {/* Log */}
      {log.length > 0 && <ProgressLog lines={log} />}

      {/* Results tabs */}
      {resultSessionId && (
        <div className="bg-panel border border-accent/30 rounded overflow-hidden">
          {/* Tab bar */}
          <div className="flex border-b border-accent/20">
            {[
              { id: '4view', label: '4-View Grid' },
              { id: 'telemetry', label: 'Telemetry' },
              { id: 'manifest', label: 'Manifest' },
            ].map((t) => (
              <button
                key={t.id}
                onClick={() => setActiveTab(t.id)}
                className={`px-4 py-2 text-xs border-r border-accent/20 transition-colors ${
                  activeTab === t.id
                    ? 'bg-accent/20 text-gray-200'
                    : 'text-gray-500 hover:text-gray-300'
                }`}
              >
                {t.label}
              </button>
            ))}
          </div>

          <div className="p-4">
            {activeTab === '4view' && (
              <div className="flex flex-col gap-3">
                <FourViewGrid sessionId={resultSessionId} frameIndex={frameIndex} />
                <input
                  type="range" min={0} max={99} value={frameIndex}
                  onChange={(e) => setFrameIndex(Number(e.target.value))}
                  className="w-full accent-accent"
                />
                <p className="text-xs text-gray-500 text-center">Frame {frameIndex}</p>
              </div>
            )}
            {activeTab === 'telemetry' && (
              <TelemetryPanel sessionId={resultSessionId} />
            )}
            {activeTab === 'manifest' && (
              <ManifestPanel sessionId={resultSessionId} />
            )}
          </div>
        </div>
      )}

      {/* Summary stats */}
      {results && (
        <div className="bg-panel border border-green-700/40 rounded p-4 grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            ['Frames / View', results.total_frames_per_view],
            ['GPS Samples', results.gps_samples],
            ['IMU Samples', results.imu_samples],
            ['Disk Used (GB)', results.disk_usage_gb?.toFixed(2)],
            ['Processing Time (min)', results.processing_time_minutes?.toFixed(1)],
            ['IMU Rate (Hz)', results.imu_rate_hz?.toFixed(1)],
            ['GPS Rate (Hz)', results.gps_rate_hz?.toFixed(1)],
            ['Session ID', results.session_id],
          ].map(([label, val]) => (
            <div key={label} className="flex flex-col gap-0.5">
              <span className="text-[10px] text-gray-500">{label}</span>
              <span className="text-sm text-gray-200 font-mono">{val ?? '—'}</span>
            </div>
          ))}
          {results.issues?.length > 0 && (
            <div className="col-span-2 sm:col-span-4">
              <p className="text-xs text-yellow-400">
                Issues: {results.issues.join(' · ')}
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
