import { useState } from 'react'
import { runTraining, cancelTraining, exportModel, connectTaskWS } from '../api/client'
import ProgressLog from '../components/ProgressLog'
import { LineChart, Line, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from 'recharts'

const DEFAULT_CFG = {
  dataset_dir: '',
  model_variant: 'yolov8n',
  pretrained_weights: 'yolov8n.pt',
  epochs: 100,
  batch_size: 16,
  image_size: 640,
  learning_rate: 0.01,
  device: 'auto',
  project_name: 'rover_detection',
  run_name: 'run1',
}

export default function Training({ sessionId }) {
  const [cfg, setCfg]           = useState(DEFAULT_CFG)
  const [log, setLog]           = useState([])
  const [progress, setProgress] = useState(null)
  const [running, setRunning]   = useState(false)
  const [taskId, setTaskId]     = useState(null)
  const [epochData, setEpochData] = useState([])
  const [trainResult, setTrainResult] = useState(null)
  const [exportFmt, setExportFmt]     = useState('onnx')
  const [exported, setExported]       = useState(null)
  const [error, setError]       = useState(null)

  async function handleRun() {
    if (!cfg.dataset_dir) { setError('Dataset directory is required.'); return }
    setRunning(true); setError(null); setLog([]); setProgress(0);
    setEpochData([]); setTrainResult(null); setExported(null)
    try {
      const { task_id } = await runTraining(cfg)
      setTaskId(task_id)
      const ws = connectTaskWS(task_id, (msg) => {
        if (msg.message) {
          // Check for epoch data embedded in message
          try {
            const inner = JSON.parse(msg.message)
            if (inner.type === 'epoch') {
              setEpochData((prev) => [...prev, inner])
              return
            }
          } catch (_) {}
          setLog((l) => [...l, msg.message])
        }
        if (msg.progress >= 0) setProgress(msg.progress)
        if (msg.status === 'done' || msg.status === 'failed') {
          setRunning(false); ws.close()
          if (msg.status === 'done') setTrainResult(msg)
        }
      })
    } catch (e) { setError(e.message); setRunning(false) }
  }

  async function handleCancel() {
    if (!taskId) return
    try { await cancelTraining({ task_id: taskId }) } catch (_) {}
    setRunning(false)
  }

  async function handleExport() {
    if (!trainResult?.best_weights_path) return
    try {
      const { exported_path } = await exportModel({
        weights_path: trainResult.best_weights_path,
        format: exportFmt,
        image_size: cfg.image_size,
      })
      setExported(exported_path)
    } catch (e) { setError(e.message) }
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
                     focus:outline-none focus:border-highlight"
        />
      </label>
    )
  }

  const lossData = epochData.map((e) => ({
    epoch: e.epoch, box: e.box_loss, cls: e.cls_loss, dfl: e.dfl_loss,
  }))
  const mapData = epochData.map((e) => ({
    epoch: e.epoch, map50: e.map50, map5095: e.map50_95,
  }))

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Model Training</h2>

      {/* Config */}
      <div className="bg-panel border border-accent rounded p-4 grid grid-cols-3 gap-3">
        <label className="flex flex-col gap-1 text-xs text-muted col-span-3">
          Dataset directory (YOLO format)
          <input
            value={cfg.dataset_dir}
            onChange={(e) => setCfg((c) => ({ ...c, dataset_dir: e.target.value }))}
            placeholder="/path/to/augmented_dataset"
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          />
        </label>
        <label className="flex flex-col gap-1 text-xs text-muted">
          Base model
          <select
            value={cfg.model_variant}
            onChange={(e) => setCfg((c) => ({ ...c, model_variant: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {['yolov8n','yolov8s','yolov8m','yolov8l','yolov8x'].map((m) => <option key={m}>{m}</option>)}
          </select>
        </label>
        {field('epochs',        'Epochs',       'number', { min: 1, max: 2000 })}
        {field('batch_size',    'Batch size',   'number', { min: 1, max: 128 })}
        <label className="flex flex-col gap-1 text-xs text-muted">
          Image size
          <select
            value={cfg.image_size}
            onChange={(e) => setCfg((c) => ({ ...c, image_size: Number(e.target.value) }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {[416,512,640,1024].map((s) => <option key={s}>{s}</option>)}
          </select>
        </label>
        {field('learning_rate', 'Learning rate','number', { min: 1e-5, max: 0.1, step: 0.001 })}
        <label className="flex flex-col gap-1 text-xs text-muted">
          Device
          <select
            value={cfg.device}
            onChange={(e) => setCfg((c) => ({ ...c, device: e.target.value }))}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          >
            {['auto','cpu','cuda:0'].map((d) => <option key={d}>{d}</option>)}
          </select>
        </label>
        {field('project_name', 'Project name')}
        {field('run_name',     'Run name')}
      </div>

      <div className="flex gap-2">
        <button
          onClick={handleRun}
          disabled={running}
          className="px-4 py-2 bg-highlight text-white rounded text-sm font-bold
                     disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90"
        >
          {running ? 'Training…' : 'Start Training'}
        </button>
        {running && (
          <button
            onClick={handleCancel}
            className="px-4 py-2 bg-accent text-gray-300 rounded text-sm hover:bg-red-700"
          >
            Cancel
          </button>
        )}
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}

      {progress !== null && (
        <div className="bg-panel border border-accent rounded p-1">
          <div className="h-2 bg-bg rounded overflow-hidden">
            <div className="h-2 bg-highlight rounded transition-all" style={{ width: `${progress}%` }} />
          </div>
        </div>
      )}

      {/* Live charts */}
      {epochData.length > 0 && (
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-panel border border-accent rounded p-3">
            <p className="text-xs text-muted mb-2">Losses</p>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={lossData}>
                <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: '#666680' }} />
                <YAxis tick={{ fontSize: 9, fill: '#666680' }} />
                <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #0f3460', fontSize: 10 }} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="box" stroke="#e94560" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="cls" stroke="#4fc3f7" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="dfl" stroke="#66bb6a" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="bg-panel border border-accent rounded p-3">
            <p className="text-xs text-muted mb-2">mAP</p>
            <ResponsiveContainer width="100%" height={160}>
              <LineChart data={mapData}>
                <XAxis dataKey="epoch" tick={{ fontSize: 9, fill: '#666680' }} />
                <YAxis tick={{ fontSize: 9, fill: '#666680' }} domain={[0, 1]} />
                <Tooltip contentStyle={{ background: '#16213e', border: '1px solid #0f3460', fontSize: 10 }} />
                <Legend wrapperStyle={{ fontSize: 10 }} />
                <Line type="monotone" dataKey="map50" stroke="#e94560" dot={false} strokeWidth={2} />
                <Line type="monotone" dataKey="map5095" stroke="#4fc3f7" dot={false} strokeWidth={2} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      )}

      <ProgressLog lines={log} />

      {/* Results + export */}
      {trainResult && (
        <div className="bg-panel border border-accent rounded p-4">
          <p className="text-gray-400 font-bold text-sm mb-3">Training Complete</p>
          <div className="grid grid-cols-3 gap-2 text-xs mb-4">
            <div><span className="text-muted">Best mAP50: </span><span className="text-green-400">{trainResult.final_map50?.toFixed(4)}</span></div>
            <div><span className="text-muted">Best epoch: </span><span>{trainResult.best_epoch}</span></div>
            <div><span className="text-muted">Weights: </span><span className="font-mono">{trainResult.best_weights_path}</span></div>
          </div>
          <div className="flex gap-2 items-center">
            <select
              value={exportFmt}
              onChange={(e) => setExportFmt(e.target.value)}
              className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
            >
              {['onnx','torchscript','tflite'].map((f) => <option key={f}>{f}</option>)}
            </select>
            <button
              onClick={handleExport}
              className="px-3 py-1.5 bg-accent text-gray-300 rounded text-sm hover:bg-highlight/70"
            >
              Export Model
            </button>
            {exported && <span className="text-green-400 text-xs">Exported: {exported}</span>}
          </div>
        </div>
      )}
    </div>
  )
}
