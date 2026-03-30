const COLOR = {
  pending: 'bg-gray-600 text-gray-200',
  running: 'bg-yellow-600 text-yellow-100',
  done:    'bg-green-700 text-green-100',
  failed:  'bg-red-700 text-red-100',
}

export default function StatusBadge({ status }) {
  const cls = COLOR[status] ?? 'bg-gray-700 text-gray-300'
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-xs font-semibold ${cls}`}>
      {status}
    </span>
  )
}
