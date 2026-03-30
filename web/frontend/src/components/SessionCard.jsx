import StatusBadge from './StatusBadge'

export default function SessionCard({ session, selected, onClick, onDelete }) {
  return (
    <div
      onClick={onClick}
      className={`p-3 rounded border cursor-pointer transition-colors
        ${selected
          ? 'border-highlight bg-accent'
          : 'border-accent bg-panel hover:border-highlight'}`}
    >
      <div className="flex items-center justify-between gap-2">
        <span className="font-semibold text-sm truncate">{session.name}</span>
        <StatusBadge status={session.extraction_status} />
      </div>
      <div className="text-xs text-muted mt-1">
        {session.environment} · {session.location}
      </div>
      <div className="text-xs text-muted">
        {session.files?.length ?? 0} file(s) · {session.audit_results_count ?? 0} audited
      </div>
      {onDelete && (
        <button
          className="mt-2 text-xs text-red-400 hover:text-red-300"
          onClick={(e) => { e.stopPropagation(); onDelete(session.id) }}
        >
          Delete
        </button>
      )}
    </div>
  )
}
