import { useState, useEffect } from 'react'
import {
  getVersioningHealth, listVersions, commitVersion,
  diffVersions, restoreVersion,
} from '../api/client'

export default function Versioning({ sessionId }) {
  const [datasetDir, setDatasetDir] = useState('')
  const [health, setHealth]         = useState(null)
  const [versions, setVersions]     = useState([])
  const [selected, setSelected]     = useState(null)
  const [diffA, setDiffA]           = useState('')
  const [diffB, setDiffB]           = useState('')
  const [diffResult, setDiffResult] = useState(null)
  const [commitTag, setCommitTag]   = useState('')
  const [commitMsg, setCommitMsg]   = useState('')
  const [error, setError]           = useState(null)
  const [status, setStatus]         = useState('')

  useEffect(() => {
    getVersioningHealth().then(setHealth).catch(() => {})
  }, [])

  async function loadVersions() {
    if (!datasetDir.trim()) return
    setError(null)
    try {
      const v = await listVersions(datasetDir)
      setVersions(v)
    } catch (e) { setError(e.message) }
  }

  async function handleCommit() {
    if (!datasetDir || !commitTag || !commitMsg) return
    setError(null)
    try {
      await commitVersion({ dataset_dir: datasetDir, version_tag: commitTag, message: commitMsg })
      setStatus(`✓ Version ${commitTag} committed`)
      setCommitTag(''); setCommitMsg('')
      loadVersions()
    } catch (e) { setError(e.message) }
  }

  async function handleDiff() {
    if (!datasetDir || !diffA || !diffB) return
    setError(null)
    try {
      const d = await diffVersions({ dataset_dir: datasetDir, version_a: diffA, version_b: diffB })
      setDiffResult(d)
    } catch (e) { setError(e.message) }
  }

  async function handleRestore(tag) {
    if (!window.confirm(`Restore dataset to version "${tag}"? Uncommitted changes will be lost.`)) return
    setError(null)
    try {
      await restoreVersion(datasetDir, tag)
      setStatus(`✓ Restored to ${tag}`)
    } catch (e) { setError(e.message) }
  }

  return (
    <div className="flex flex-col gap-4">
      <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Dataset Versioning (DVC)</h2>

      {health && (
        <div className="flex gap-3">
          <span className={`px-2 py-1 rounded text-xs ${health.dvc ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'}`}>
            {health.dvc ? '✓ DVC Ready' : '⚠ DVC not installed'}
          </span>
        </div>
      )}

      {!health?.dvc && (
        <div className="bg-yellow-900/30 border border-yellow-700 rounded p-3 text-xs text-yellow-300">
          Install DVC: <code>pip install dvc</code>. Also requires git to be initialized in dataset directory.
        </div>
      )}

      <div className="bg-panel border border-accent rounded p-4 flex gap-3 items-end">
        <label className="flex flex-col gap-1 text-xs text-muted flex-1">
          Dataset directory
          <input type="text" placeholder="/path/to/dataset"
            value={datasetDir}
            onChange={(e) => setDatasetDir(e.target.value)}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300"
          />
        </label>
        <button onClick={loadVersions}
          className="px-3 py-1.5 bg-accent text-gray-300 rounded text-sm hover:bg-hover">
          Load versions
        </button>
      </div>

      {error && <p className="text-red-400 text-sm">{error}</p>}
      {status && <p className="text-green-400 text-sm">{status}</p>}

      {versions.length > 0 && (
        <div className="bg-panel border border-accent rounded p-4">
          <p className="text-xs font-bold text-gray-400 mb-2">Versions ({versions.length})</p>
          <div className="flex flex-col gap-1 max-h-64 overflow-y-auto">
            {versions.map((v) => (
              <button key={v.tag}
                onClick={() => setSelected(v === selected ? null : v)}
                className={`text-left px-3 py-2 rounded text-xs flex justify-between items-center
                  ${selected?.tag === v.tag ? 'bg-accent text-white' : 'hover:bg-hover text-gray-300'}`}
              >
                <span className="font-mono">{v.tag}</span>
                <span className="text-muted">{v.total_frames} frames · {new Date(v.timestamp).toLocaleDateString()}</span>
              </button>
            ))}
          </div>

          {selected && (
            <div className="mt-3 p-3 bg-bg rounded text-xs text-gray-300 space-y-1">
              <p><span className="text-muted">Message: </span>{selected.message}</p>
              <p><span className="text-muted">Files: </span>{selected.file_count}</p>
              <p><span className="text-muted">Hash: </span><code className="text-muted">{selected.dataset_hash?.slice(0, 12)}…</code></p>
              {selected.class_distribution && (
                <div className="flex flex-wrap gap-1 mt-1">
                  {Object.entries(selected.class_distribution).map(([k, v]) => (
                    <span key={k} className="bg-accent px-1.5 py-0.5 rounded">{k}: {v}</span>
                  ))}
                </div>
              )}
              <button onClick={() => handleRestore(selected.tag)}
                className="mt-2 px-3 py-1 bg-red-800 text-red-200 rounded text-xs hover:bg-red-700">
                Restore this version
              </button>
            </div>
          )}
        </div>
      )}

      <div className="bg-panel border border-accent rounded p-4 flex flex-col gap-3">
        <p className="text-xs font-bold text-gray-400">Commit new version</p>
        <div className="flex gap-3">
          <input type="text" placeholder="v1.0.0" value={commitTag}
            onChange={(e) => setCommitTag(e.target.value)}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300 w-28"
          />
          <input type="text" placeholder="Added highway session data" value={commitMsg}
            onChange={(e) => setCommitMsg(e.target.value)}
            className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300 flex-1"
          />
          <button onClick={handleCommit}
            className="px-3 py-1.5 bg-green-800 text-green-200 rounded text-sm hover:bg-green-700">
            Commit
          </button>
        </div>
      </div>

      {versions.length >= 2 && (
        <div className="bg-panel border border-accent rounded p-4 flex flex-col gap-3">
          <p className="text-xs font-bold text-gray-400">Compare versions</p>
          <div className="flex gap-3 items-end">
            <select value={diffA} onChange={(e) => setDiffA(e.target.value)}
              className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300">
              <option value="">Version A</option>
              {versions.map((v) => <option key={v.tag} value={v.tag}>{v.tag}</option>)}
            </select>
            <select value={diffB} onChange={(e) => setDiffB(e.target.value)}
              className="bg-bg border border-accent rounded px-2 py-1 text-sm text-gray-300">
              <option value="">Version B</option>
              {versions.map((v) => <option key={v.tag} value={v.tag}>{v.tag}</option>)}
            </select>
            <button onClick={handleDiff}
              className="px-3 py-1.5 bg-accent text-gray-300 rounded text-sm hover:bg-hover">
              Compare
            </button>
          </div>
          {diffResult && (
            <div className="text-xs text-gray-300 grid grid-cols-2 gap-2 mt-1">
              <div><span className="text-muted">Frames added: </span><span className="text-green-400">{diffResult.frames_added}</span></div>
              <div><span className="text-muted">Frames removed: </span><span className="text-red-400">{diffResult.frames_removed}</span></div>
              <div><span className="text-muted">Total delta: </span><span>{diffResult.total_frames_delta > 0 ? '+' : ''}{diffResult.total_frames_delta}</span></div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
