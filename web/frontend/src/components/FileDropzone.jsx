/**
 * FileDropzone — drag-and-drop file path input.
 * Note: browsers can't read actual file paths for security reasons.
 * This component accepts typed paths (for server-side files) or
 * displays dropped filenames as confirmation (actual paths are entered manually).
 */
import { useState, useRef } from 'react'

export default function FileDropzone({ onPaths, accept = '.mp4,.insv,.csv,.json' }) {
  const [dragging, setDragging] = useState(false)
  const [paths, setPaths] = useState([])
  const inputRef = useRef(null)

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    const names = Array.from(e.dataTransfer.files).map((f) => f.name)
    setPaths(names)
    // Inform parent — these are filenames only, not full paths
    onPaths?.(names)
  }

  function handleManualInput(e) {
    const val = e.target.value.trim()
    if (!val) return
    const list = val.split('\n').map((s) => s.trim()).filter(Boolean)
    setPaths(list)
    onPaths?.(list)
  }

  return (
    <div
      className={`border-2 border-dashed rounded p-4 transition-colors cursor-pointer
        ${dragging ? 'border-highlight bg-accent/30' : 'border-accent hover:border-highlight'}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => inputRef.current?.focus()}
    >
      <p className="text-muted text-sm text-center mb-2">
        Drop files here, or enter server-side paths below
      </p>
      <textarea
        ref={inputRef}
        rows={3}
        placeholder="/data/sessions/GH010001.MP4"
        className="w-full bg-bg border border-accent rounded p-2 text-sm font-mono text-gray-300
                   focus:outline-none focus:border-highlight resize-none"
        onChange={handleManualInput}
      />
      {paths.length > 0 && (
        <ul className="mt-2 text-xs text-muted space-y-0.5">
          {paths.map((p, i) => <li key={i} className="truncate">• {p}</li>)}
        </ul>
      )}
    </div>
  )
}
