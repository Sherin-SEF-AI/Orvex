import { useState } from 'react'
import { runAugmentation, getAugmentationResults, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const DEFAULT_CFG = {
  horizontal_flip: true,
  vertical_flip: false,
  random_rotate_90: true,
  brightness_contrast: true,
  hue_saturation: true,
  gaussian_noise: true,
  motion_blur: true,
  jpeg_compression: true,
  mosaic: true,
  rain_simulation: false,
  fog_simulation: false,
  multiplier: 3,
}

const GEO_TRANSFORMS = [
  ['horizontal_flip',  'Horizontal flip'],
  ['vertical_flip',    'Vertical flip'],
  ['random_rotate_90', 'Random rotate 90°'],
]
const PHOTO_TRANSFORMS = [
  ['brightness_contrast', 'Brightness / Contrast'],
  ['hue_saturation',      'Hue / Saturation'],
  ['gaussian_noise',      'Gaussian noise'],
  ['motion_blur',         'Motion blur'],
  ['jpeg_compression',    'JPEG compression'],
]
const ADV_TRANSFORMS = [
  ['mosaic',          'Mosaic (YOLOv8 style)'],
  ['rain_simulation', 'Rain simulation'],
  ['fog_simulation',  'Fog simulation'],
]

export default function Augmentation({ sessionId }) {
  const [cfg, setCfg]           = useState(DEFAULT_CFG)
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [result, setResult]     = useState(null)
  const [error, setError]       = useState(null)

  if (!sessionId) return <p className="text-muted text-sm">Select a session first.</p>

  function toggleCheck(key) {
    setCfg((c) => ({ ...c, [key]: !c[key] }))
  }

  async function handleRun() {
    setRunning(true); setError(null); setLog([]); setProgress(0); setResult(null)
    try {
      const { task_id } = await runAugmentation(sessionId, cfg)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') {
            getAugmentationResults(sessionId).then((d) => setResult(d.result)).catch(() => {})
          }
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  function TransformGroup({ title, transforms }) {
    return (
      <div className="bg-panel border border-accent rounded p-3">
        <p className="text-xs text-muted mb-2 font-bold">{title}</p>
        <div className="flex flex-col gap-1.5">
          {transforms.map(([key, label]) => (
            <label key={key} className="flex items-center gap-2 text-xs text-gray-300 cursor-pointer">
              <input
                type="checkbox"
                checked={cfg[key]}
                onChange={() => toggleCheck(key)}
                className="accent-highlight"
              />
              {label}
            </label>
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Data Augmentation</h2>

      <div className="bg-panel border border-accent rounded p-3 text-xs text-muted">
        Requires auto-labeling to have been run first. Outputs YOLO-format dataset
        to <code className="text-gray-300">augmented_dataset/</code>.
      </div>

      <div className="grid grid-cols-3 gap-3">
        <TransformGroup title="Geometric" transforms={GEO_TRANSFORMS} />
        <TransformGroup title="Photometric" transforms={PHOTO_TRANSFORMS} />
        <TransformGroup title="Advanced" transforms={ADV_TRANSFORMS} />
      </div>

      <div className="bg-panel border border-accent rounded p-3 flex items-center gap-3">
        <label className="text-xs text-muted">Dataset multiplier</label>
        <input
          type="number" min={2} max={20}
          value={cfg.multiplier}
          onChange={(e) => setCfg((c) => ({ ...c, multiplier: Number(e.target.value) }))}
          className="bg-bg border border-accent rounded px-2 py-1 text-sm w-20 text-gray-300"
        />
        <span className="text-xs text-muted">Output = N_original × multiplier images</span>
      </div>

      <button
        onClick={handleRun}
        disabled={running}
        className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                   disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
      >
        {running ? 'Augmenting…' : 'Augment Dataset'}
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
        <div className="bg-panel border border-accent rounded p-4 text-sm">
          <p className="text-gray-400 font-bold mb-2">Augmentation Results</p>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div><span className="text-muted">Original images: </span><span>{result.original_count}</span></div>
            <div><span className="text-muted">Augmented total: </span><span className="text-green-400">{result.augmented_count}</span></div>
            <div className="col-span-2"><span className="text-muted">Output dir: </span><span className="text-gray-300 font-mono">{result.output_dir}</span></div>
          </div>
        </div>
      )}
    </div>
  )
}
