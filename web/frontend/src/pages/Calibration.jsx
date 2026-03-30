import { useState } from 'react'
import { createCalibration, runCalibStep, getCalibSteps, getCalibrationHealth } from '../api/client'
import { connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'

const STEPS = [
  {
    key:  'imu_static',
    label: 'Step 1 — IMU Static',
    desc:  'Place GoPro flat on a table. Record for 4+ hours with the lens cap on. '
         + 'Enter the path to the MP4 below.',
  },
  {
    key:  'camera_intrinsic',
    label: 'Step 2 — Camera Intrinsic',
    desc:  'Move the GoPro in front of an AprilTag board, covering all corners of the frame. '
         + 'Enter the path to the calibration video MP4.',
  },
  {
    key:  'camera_imu_extrinsic',
    label: 'Step 3 — Camera-IMU Extrinsic',
    desc:  'Shake the GoPro aggressively in front of a fixed AprilTag board for 2–3 minutes. '
         + 'Requires Step 1 results. Enter the path to the extrinsic calibration video.',
  },
]

export default function Calibration() {
  const [calId, setCalId]       = useState(null)
  const [device, setDevice]     = useState('gopro')
  const [currentStep, setStep]  = useState(0)
  const [filePath, setFilePath] = useState('')
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [stepResults, setStepResults] = useState({})
  const [healthModal, setHealthModal] = useState(null)   // null | { ok, checks }
  const [error, setError]       = useState(null)

  async function handleHealthCheck() {
    if (!calId) { setError('Run at least one calibration step first.'); return }
    try {
      const result = await getCalibrationHealth(calId)
      setHealthModal(result)
    } catch (e) { setError(e.message) }
  }

  async function handleStart() {
    setError(null)
    try {
      const cal = await createCalibration({
        camera_device: device,
        session_type: STEPS[currentStep].key,
        file_path: filePath,
      })
      setCalId(cal.id)
    } catch (e) { setError(e.message) }
  }

  async function handleRunStep() {
    if (!calId) { await handleStart(); return }
    setRunning(true); setError(null); setLog([]); setProgress(0)
    try {
      const { task_id } = await runCalibStep(calId, {
        step: STEPS[currentStep].key,
        file_path: filePath,
        extra: {},
      })
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) setLog((l) => [...l, msg.message])
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          getCalibSteps(calId).then(setStepResults).catch(() => {})
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  const step = STEPS[currentStep]

  return (
    <div className="flex flex-col gap-4 max-w-2xl">
      <div className="flex items-center gap-3">
        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide flex-1">Calibration</h2>
        <button onClick={handleHealthCheck} disabled={!calId}
          className="text-xs px-3 py-1 border border-accent rounded hover:border-highlight
                     disabled:opacity-40 transition-colors">
          Health Check
        </button>
      </div>

      {/* Health Check Modal */}
      {healthModal && (
        <div className="fixed inset-0 bg-black/60 flex items-center justify-center z-50"
             onClick={() => setHealthModal(null)}>
          <div className="bg-[#16213e] border border-accent rounded-lg p-6 w-96 max-w-full"
               onClick={(e) => e.stopPropagation()}>
            <div className="flex items-center gap-2 mb-4">
              <span className={`text-lg font-bold ${healthModal.ok ? 'text-green-400' : 'text-red-400'}`}>
                {healthModal.ok ? '✓ Calibration Healthy' : '✕ Issues Detected'}
              </span>
            </div>
            <table className="w-full text-xs border-collapse">
              <thead>
                <tr className="text-muted border-b border-accent">
                  <th className="text-left px-2 py-1">Check</th>
                  <th className="text-center px-2 py-1">Result</th>
                  <th className="text-left px-2 py-1">Detail</th>
                </tr>
              </thead>
              <tbody>
                {healthModal.checks?.map((c, i) => (
                  <tr key={i} className="border-b border-accent/30">
                    <td className="px-2 py-1">{c.name}</td>
                    <td className={`px-2 py-1 text-center font-bold ${c.passed ? 'text-green-400' : 'text-red-400'}`}>
                      {c.passed ? '✓' : '✕'}
                    </td>
                    <td className="px-2 py-1 text-muted">{c.detail}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button onClick={() => setHealthModal(null)}
              className="mt-4 w-full py-1.5 text-xs bg-accent rounded hover:bg-highlight transition-colors">
              Close
            </button>
          </div>
        </div>
      )}

      {/* Stepper */}
      <div className="flex gap-2">
        {STEPS.map((s, i) => {
          const res = stepResults[s.key]
          const icon = running && i === currentStep ? '◉'
            : res?.failed ? '✕'
            : res?.complete ? '●'
            : i === currentStep ? '◉'
            : '○'
          return (
            <button key={s.key} onClick={() => setStep(i)}
              className={`flex-1 py-2 text-xs rounded border transition-colors
                ${i === currentStep
                  ? 'border-highlight bg-accent text-white font-semibold'
                  : res?.complete
                    ? 'border-green-600 text-green-400'
                    : res?.failed
                      ? 'border-red-600 text-red-400'
                      : 'border-accent text-muted hover:border-highlight'}`}>
              <span className="mr-1">{icon}</span>
              {s.label.split('—')[1]?.trim()}
            </button>
          )
        })}
      </div>

      {/* Step instructions */}
      <div className="bg-panel border border-accent rounded p-4 text-sm text-gray-300 leading-relaxed">
        <p className="font-semibold text-white mb-2">{step.label}</p>
        <p>{step.desc}</p>
      </div>

      {/* Device selector (step 0 only) */}
      {currentStep === 0 && (
        <label className="flex items-center gap-3 text-xs text-muted">
          Camera device:
          <select
            value={device}
            onChange={(e) => setDevice(e.target.value)}
            className="bg-panel border border-accent rounded px-2 py-1 text-sm text-gray-300
                       focus:outline-none focus:border-highlight"
          >
            <option value="gopro">GoPro Hero 11</option>
            <option value="insta360">Insta360 X4</option>
          </select>
        </label>
      )}

      {/* File path input */}
      <input
        value={filePath}
        onChange={(e) => setFilePath(e.target.value)}
        placeholder="/path/to/calibration_video.MP4"
        className="bg-panel border border-accent rounded px-3 py-2 text-sm font-mono
                   focus:outline-none focus:border-highlight"
      />

      {error && <p className="text-red-400 text-xs">{error}</p>}

      <button onClick={handleRunStep} disabled={running || !filePath}
        className="self-start text-sm px-5 py-2 bg-green-700 hover:bg-green-600 text-white
                   rounded font-semibold disabled:opacity-50 transition-colors">
        {running ? 'Running…' : `Run ${step.label.split('—')[1]?.trim()}`}
      </button>

      {(running || log.length > 0) && <ProgressLog lines={log} progress={progress} />}

      {/* Step results */}
      {Object.entries(stepResults).map(([k, v]) => v?.complete && (
        <div key={k} className="bg-panel border border-green-700 rounded p-3 text-xs">
          <p className="text-green-400 font-semibold mb-1">{k} ✓</p>
          <pre className="text-gray-400 overflow-x-auto">
            {JSON.stringify(v.result, null, 2)}
          </pre>
        </div>
      ))}
    </div>
  )
}
