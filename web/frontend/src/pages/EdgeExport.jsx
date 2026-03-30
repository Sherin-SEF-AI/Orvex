import { useState, useEffect } from 'react'
import { checkExportDependencies, exportONNX, exportTRT, benchmarkModel, buildJetsonPackage } from '../api/client'

export default function EdgeExport() {
  const [deps, setDeps]         = useState(null)
  const [step, setStep]         = useState(0)
  const [weightsPath, setWeightsPath] = useState('')
  const [onnxResult, setOnnxResult]   = useState(null)
  const [trtResult, setTrtResult]     = useState(null)
  const [benchResults, setBenchResults] = useState([])
  const [packageResult, setPackageResult] = useState(null)
  const [running, setRunning]   = useState(false)
  const [error, setError]       = useState(null)
  const [status, setStatus]     = useState('')

  const [onnxCfg, setOnnxCfg]   = useState({ image_size: 640, batch_size: 1, simplify: true, opset_version: 17 })
  const [trtCfg, setTrtCfg]     = useState({ precision: 'fp16', workspace_gb: 4 })
  const [benchFmt, setBenchFmt] = useState('onnx')
  const [pkgCfg, setPkgCfg]     = useState({ target_device: 'jetson_orin', conf_threshold: 0.25, class_names: 'car,truck,person,motorcycle,bicycle', output_dir: '' })

  useEffect(() => {
    checkExportDependencies().then(setDeps).catch(() => {})
  }, [])

  const steps = ['ONNX Export', 'TensorRT', 'Benchmark', 'Package']

  async function doONNX() {
    if (!weightsPath) return
    setRunning(true); setError(null)
    try {
      const r = await exportONNX({ weights_path: weightsPath, output_path: weightsPath.replace('.pt', '.onnx'), ...onnxCfg })
      setOnnxResult(r)
      setStatus('✓ ONNX export complete')
    } catch (e) { setError(e.message) } finally { setRunning(false) }
  }

  async function doTRT() {
    if (!onnxResult?.output_path) return
    setRunning(true); setError(null)
    try {
      const r = await exportTRT({ onnx_path: onnxResult.output_path, output_path: onnxResult.output_path.replace('.onnx', '.engine'), ...trtCfg })
      setTrtResult(r)
      setStatus('✓ TensorRT conversion complete')
    } catch (e) { setError(e.message) } finally { setRunning(false) }
  }

  async function doBenchmark() {
    const modelPath = benchFmt === 'onnx' ? onnxResult?.output_path : weightsPath
    if (!modelPath) return
    setRunning(true); setError(null)
    try {
      const r = await benchmarkModel({ model_path: modelPath, model_format: benchFmt, image_size: onnxCfg.image_size })
      setBenchResults((prev) => {
        const filtered = prev.filter((x) => x.format !== r.format)
        return [...filtered, r]
      })
      setStatus(`✓ Benchmark done: ${r.throughput_fps.toFixed(1)} FPS`)
    } catch (e) { setError(e.message) } finally { setRunning(false) }
  }

  async function doPackage() {
    if (!onnxResult?.output_path || !pkgCfg.output_dir) return
    setRunning(true); setError(null)
    try {
      const classNames = pkgCfg.class_names.split(',').map((s) => s.trim()).filter(Boolean)
      const r = await buildJetsonPackage({
        weights_path: weightsPath,
        onnx_path: onnxResult.output_path,
        trt_path: trtResult?.output_path || null,
        class_names: classNames,
        output_dir: pkgCfg.output_dir,
        target_device: pkgCfg.target_device,
        conf_threshold: pkgCfg.conf_threshold,
      })
      setPackageResult(r)
      setStatus(`✓ Package built: ${r.tar_path}`)
    } catch (e) { setError(e.message) } finally { setRunning(false) }
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Edge Export — Jetson / ONNX / TensorRT</h2>

      {deps && (
        <div className="flex flex-wrap gap-2">
          {Object.entries(deps).map(([k, v]) => (
            <span key={k} className={`px-2 py-0.5 rounded text-xs ${v === true ? 'bg-green-900 text-green-300' : v === false ? 'bg-gray-800 text-gray-500' : 'bg-accent text-gray-300'}`}>
              {k}: {String(v)}
            </span>
          ))}
        </div>
      )}

      <div className="bg-panel border border-accent rounded p-4 flex gap-3 items-end">
        <label className="flex flex-col gap-1 text-xs text-muted flex-1">
          Model weights (.pt)
          <input type="text" placeholder="/path/to/best.pt"
            value={weightsPath}
            onChange={(e) => setWeightsPath(e.target.value)}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          />
        </label>
      </div>

      {/* Step tabs */}
      <div className="flex gap-1">
        {steps.map((s, i) => (
          <button key={s} onClick={() => setStep(i)}
            className={`px-3 py-1.5 rounded text-xs font-bold
              ${step === i ? 'bg-highlight text-white' : 'bg-panel border border-accent text-gray-400 hover:bg-hover'}`}>
            {i + 1}. {s}
          </button>
        ))}
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}
      {status && <p className="text-green-400 text-sm">{status}</p>}

      {/* Step 0: ONNX */}
      {step === 0 && (
        <div className="bg-panel border border-accent rounded p-4 flex flex-col gap-3">
          <p className="text-xs font-bold text-gray-400">ONNX Export</p>
          <div className="flex flex-wrap gap-3">
            <label className="flex flex-col gap-1 text-xs text-muted">
              Image size
              <select value={onnxCfg.image_size} onChange={(e) => setOnnxCfg((c) => ({ ...c, image_size: Number(e.target.value) }))}
                className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300">
                {[416, 512, 640, 1280].map((s) => <option key={s}>{s}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-muted">
              Opset
              <input type="number" min={11} max={17} value={onnxCfg.opset_version}
                onChange={(e) => setOnnxCfg((c) => ({ ...c, opset_version: Number(e.target.value) }))}
                className="bg-bg border border-accent rounded px-2 py-1 text-sm w-16 text-gray-300" />
            </label>
            <label className="flex items-center gap-2 text-xs text-muted mt-4">
              <input type="checkbox" checked={onnxCfg.simplify} onChange={(e) => setOnnxCfg((c) => ({ ...c, simplify: e.target.checked }))} />
              Simplify
            </label>
          </div>
          <button onClick={doONNX} disabled={running || !weightsPath}
            className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold disabled:opacity-50">
            {running ? 'Exporting…' : 'Export ONNX'}
          </button>
          {onnxResult && (
            <div className="text-xs text-green-400 space-y-1">
              <p>✓ {onnxResult.output_path}</p>
              <p>Size: {onnxResult.model_size_mb?.toFixed(1)} MB · Latency: {onnxResult.test_latency_ms?.toFixed(1)} ms · Verified: {onnxResult.verification_passed ? '✓' : '✗'}</p>
            </div>
          )}
        </div>
      )}

      {/* Step 1: TRT */}
      {step === 1 && (
        <div className="bg-panel border border-accent rounded p-4 flex flex-col gap-3">
          <p className="text-xs font-bold text-gray-400">TensorRT Conversion</p>
          {!deps?.tensorrt && (
            <div className="bg-yellow-900/30 border border-yellow-700 rounded p-2 text-xs text-yellow-300">
              trtexec not found. On Jetson: <code>sudo apt install tensorrt</code>
            </div>
          )}
          <div className="flex gap-3">
            <label className="flex flex-col gap-1 text-xs text-muted">
              Precision
              <select value={trtCfg.precision} onChange={(e) => setTrtCfg((c) => ({ ...c, precision: e.target.value }))}
                className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300">
                {['fp32', 'fp16', 'int8'].map((p) => <option key={p}>{p}</option>)}
              </select>
            </label>
          </div>
          <button onClick={doTRT} disabled={running || !onnxResult || !deps?.tensorrt}
            className="self-start px-4 py-2 bg-highlight text-white rounded text-sm font-bold disabled:opacity-50">
            {running ? 'Converting…' : 'Convert to TRT'}
          </button>
          {trtResult && (
            <p className="text-xs text-green-400">✓ {trtResult.engine_size_mb?.toFixed(1)} MB · {trtResult.precision} · Built in {trtResult.build_time_minutes?.toFixed(1)} min</p>
          )}
        </div>
      )}

      {/* Step 2: Benchmark */}
      {step === 2 && (
        <div className="bg-panel border border-accent rounded p-4 flex flex-col gap-3">
          <p className="text-xs font-bold text-gray-400">Benchmark</p>
          <div className="flex gap-3 items-end">
            <label className="flex flex-col gap-1 text-xs text-muted">
              Format
              <select value={benchFmt} onChange={(e) => setBenchFmt(e.target.value)}
                className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300">
                <option value="pytorch">PyTorch</option>
                <option value="onnx">ONNX</option>
              </select>
            </label>
            <button onClick={doBenchmark} disabled={running}
              className="px-4 py-2 bg-highlight text-white rounded text-sm font-bold disabled:opacity-50">
              {running ? 'Benchmarking…' : 'Run Benchmark'}
            </button>
          </div>
          {benchResults.length > 0 && (
            <table className="text-xs w-full border-collapse">
              <thead><tr className="border-b border-accent text-muted">
                <th className="px-2 py-1 text-left">Format</th>
                <th className="px-2 py-1">Mean ms</th>
                <th className="px-2 py-1">P95 ms</th>
                <th className="px-2 py-1">FPS</th>
                <th className="px-2 py-1">Memory MB</th>
              </tr></thead>
              <tbody>
                {benchResults.map((r) => (
                  <tr key={r.format} className="border-b border-accent/20 text-gray-300">
                    <td className="px-2 py-1">{r.format}</td>
                    <td className="px-2 py-1 text-center">{r.mean_latency_ms?.toFixed(1)}</td>
                    <td className="px-2 py-1 text-center">{r.p95_latency_ms?.toFixed(1)}</td>
                    <td className="px-2 py-1 text-center text-green-400">{r.throughput_fps?.toFixed(1)}</td>
                    <td className="px-2 py-1 text-center">{r.memory_mb?.toFixed(0)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}

      {/* Step 3: Package */}
      {step === 3 && (
        <div className="bg-panel border border-accent rounded p-4 flex flex-col gap-3">
          <p className="text-xs font-bold text-gray-400">Package for Jetson</p>
          <div className="flex flex-wrap gap-3">
            <label className="flex flex-col gap-1 text-xs text-muted">
              Target device
              <select value={pkgCfg.target_device} onChange={(e) => setPkgCfg((c) => ({ ...c, target_device: e.target.value }))}
                className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300">
                {['jetson_orin', 'jetson_xavier', 'jetson_nano', 'cpu'].map((d) => <option key={d}>{d}</option>)}
              </select>
            </label>
            <label className="flex flex-col gap-1 text-xs text-muted w-48">
              Class names (comma-separated)
              <input type="text" value={pkgCfg.class_names}
                onChange={(e) => setPkgCfg((c) => ({ ...c, class_names: e.target.value }))}
                className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300" />
            </label>
            <label className="flex flex-col gap-1 text-xs text-muted flex-1">
              Output directory
              <input type="text" placeholder="/path/to/output"
                value={pkgCfg.output_dir}
                onChange={(e) => setPkgCfg((c) => ({ ...c, output_dir: e.target.value }))}
                className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300" />
            </label>
          </div>
          <button onClick={doPackage} disabled={running || !onnxResult || !pkgCfg.output_dir}
            className="self-start px-4 py-2 bg-green-800 text-green-200 rounded text-sm font-bold disabled:opacity-50">
            {running ? 'Building…' : 'Build Package'}
          </button>
          {packageResult && (
            <p className="text-xs text-green-400">✓ Package: {packageResult.tar_path}</p>
          )}
        </div>
      )}
    </div>
  )
}
