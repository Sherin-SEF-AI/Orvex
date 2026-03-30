import { useState, useEffect } from 'react'
import { getSessions, createSession, deleteSession } from '../api/client'
import SessionCard from '../components/SessionCard'
import FileDropzone from '../components/FileDropzone'
import { addFile } from '../api/client'

const QUALITY_STYLES = {
  excellent: 'bg-green-900 text-green-300 border border-green-700',
  good:      'bg-blue-900  text-blue-300  border border-blue-700',
  fair:      'bg-yellow-900 text-yellow-300 border border-yellow-700',
  poor:      'bg-red-900   text-red-300   border border-red-700',
}

function QualityBadge({ quality }) {
  if (!quality) return null
  return (
    <span className={`text-xs px-1.5 py-0.5 rounded font-semibold uppercase tracking-wide
                      ${QUALITY_STYLES[quality] ?? 'bg-gray-800 text-gray-400'}`}>
      {quality}
    </span>
  )
}

export default function Sessions({ onSelect, selectedId }) {
  const [sessions, setSessions] = useState([])
  const [search, setSearch]     = useState('')
  const [form, setForm]         = useState({ name: '', environment: 'road', location: '', notes: '' })
  const [creating, setCreating] = useState(false)
  const [error, setError]       = useState(null)

  async function load() {
    try { setSessions(await getSessions()) }
    catch (e) { setError(e.message) }
  }

  useEffect(() => { load() }, [])

  const filtered = sessions.filter((s) => {
    if (!search) return true
    const q = search.toLowerCase()
    return s.name.toLowerCase().includes(q) || s.location?.toLowerCase().includes(q)
  })

  async function handleCreate(e) {
    e.preventDefault()
    setCreating(true)
    setError(null)
    try {
      const s = await createSession(form)
      setSessions((prev) => [s, ...prev])
      setForm({ name: '', environment: 'road', location: '', notes: '' })
    } catch (e) { setError(e.message) }
    finally { setCreating(false) }
  }

  async function handleDelete(id) {
    if (!confirm('Delete this session?')) return
    try {
      await deleteSession(id)
      setSessions((prev) => prev.filter((s) => s.id !== id))
      if (selectedId === id) onSelect?.(null)
    } catch (e) { setError(e.message) }
  }

  return (
    <div className="flex gap-6 h-full">
      {/* Session list */}
      <div className="w-72 flex-shrink-0 flex flex-col gap-3 overflow-y-auto">
        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide">Sessions</h2>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search by name or location…"
          className="bg-panel border border-accent rounded px-3 py-1.5 text-xs
                     focus:outline-none focus:border-highlight"
        />
        {filtered.length === 0 && (
          <p className="text-muted text-sm">{sessions.length === 0 ? 'No sessions yet. Create one below.' : 'No matches.'}</p>
        )}
        {filtered.map((s) => (
          <div key={s.id} className="relative">
            <SessionCard
              session={s}
              selected={s.id === selectedId}
              onClick={() => onSelect?.(s.id)}
              onDelete={handleDelete}
            />
            {s.quality && (
              <div className="absolute top-2 right-2">
                <QualityBadge quality={s.quality} />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Create form */}
      <div className="flex-1 max-w-lg">
        <h2 className="text-sm font-bold text-gray-400 uppercase tracking-wide mb-4">New Session</h2>
        {error && <p className="text-red-400 text-sm mb-3">{error}</p>}
        <form onSubmit={handleCreate} className="flex flex-col gap-3">
          {[
            { key: 'name',        placeholder: 'Session name',      required: true  },
            { key: 'environment', placeholder: 'Environment (road, indoor…)' },
            { key: 'location',    placeholder: 'Location' },
          ].map(({ key, placeholder, required }) => (
            <input
              key={key}
              value={form[key]}
              placeholder={placeholder}
              required={required}
              onChange={(e) => setForm((f) => ({ ...f, [key]: e.target.value }))}
              className="bg-panel border border-accent rounded px-3 py-2 text-sm
                         focus:outline-none focus:border-highlight"
            />
          ))}
          <textarea
            value={form.notes}
            placeholder="Notes (optional)"
            rows={3}
            onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
            className="bg-panel border border-accent rounded px-3 py-2 text-sm
                       focus:outline-none focus:border-highlight resize-none"
          />
          <button
            type="submit"
            disabled={creating}
            className="bg-accent hover:bg-highlight text-white rounded px-4 py-2 text-sm
                       font-semibold disabled:opacity-50 transition-colors"
          >
            {creating ? 'Creating…' : 'Create Session'}
          </button>
        </form>
      </div>
    </div>
  )
}
