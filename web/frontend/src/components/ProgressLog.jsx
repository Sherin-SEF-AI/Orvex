import { useEffect, useRef } from 'react'

export default function ProgressLog({ lines, progress }) {
  const endRef = useRef(null)
  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [lines])

  return (
    <div className="flex flex-col gap-2">
      {progress != null && progress >= 0 && (
        <div className="w-full bg-panel rounded h-2 border border-accent overflow-hidden">
          <div
            className="h-full bg-green-600 transition-all"
            style={{ width: `${Math.min(100, progress)}%` }}
          />
        </div>
      )}
      <div className="bg-panel border border-accent rounded p-2 h-40 overflow-y-auto font-mono text-xs text-gray-300">
        {lines.map((l, i) => <div key={i}>{l}</div>)}
        <div ref={endRef} />
      </div>
    </div>
  )
}
