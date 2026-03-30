/**
 * TelemetryChart — displays IMU data (accel / gyro) from audit_results.
 * Uses Recharts LineChart.
 *
 * Props:
 *   samples       — list of IMUSample objects
 *   type          — 'accel' | 'gyro'
 *   motionProfile — optional list of [timestamp_ns, label] tuples
 *                   labels: 'stationary' | 'high_motion' | 'normal'
 */
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ReferenceArea,
} from 'recharts'

const ACCEL_COLORS = { ax: '#e74c3c', ay: '#2ecc71', az: '#3498db' }
const GYRO_COLORS  = { gx: '#e67e22', gy: '#9b59b6', gz: '#1abc9c' }

export default function TelemetryChart({ samples = [], type = 'accel', motionProfile = [] }) {
  const colors = type === 'accel' ? ACCEL_COLORS : GYRO_COLORS
  const keys   = type === 'accel' ? ['ax','ay','az'] : ['gx','gy','gz']
  const yLabel  = type === 'accel' ? 'm/s²' : 'rad/s'

  if (!samples.length) {
    return (
      <div className="flex items-center justify-center h-40 text-muted text-sm">
        No telemetry data loaded.
      </div>
    )
  }

  // Downsample to max 1000 points for render performance
  const step = Math.max(1, Math.floor(samples.length / 1000))
  const data = samples
    .filter((_, i) => i % step === 0)
    .map((s) => ({
      t: +((s.timestamp_ns / 1e9).toFixed(2)),
      ax: s.accel_x, ay: s.accel_y, az: s.accel_z,
      gx: s.gyro_x,  gy: s.gyro_y,  gz: s.gyro_z,
    }))

  // Build ReferenceArea bands from motion profile windows
  // motionProfile is [[timestamp_ns, label], ...] sorted by time
  const bands = []
  if (type === 'accel' && motionProfile.length > 1) {
    for (let i = 0; i < motionProfile.length - 1; i++) {
      const [tNs, label] = motionProfile[i]
      const [tNextNs]    = motionProfile[i + 1]
      if (label === 'normal') continue
      const x1 = +(tNs    / 1e9).toFixed(2)
      const x2 = +(tNextNs / 1e9).toFixed(2)
      const fill = label === 'stationary' ? 'rgba(59,130,246,0.12)' : 'rgba(239,68,68,0.12)'
      bands.push({ x1, x2, fill, label })
    }
  }

  return (
    <ResponsiveContainer width="100%" height={180}>
      <LineChart data={data} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#0f3460" />
        <XAxis dataKey="t" tick={{ fontSize: 10, fill: '#888' }} unit="s" />
        <YAxis tick={{ fontSize: 10, fill: '#888' }} unit={yLabel} width={52} />
        <Tooltip
          contentStyle={{ background: '#16213e', border: '1px solid #0f3460', fontSize: 11 }}
          labelFormatter={(v) => `${v} s`}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {bands.map((b, i) => (
          <ReferenceArea
            key={i}
            x1={b.x1}
            x2={b.x2}
            fill={b.fill}
            strokeOpacity={0}
            ifOverflow="hidden"
          />
        ))}
        {keys.map((k) => (
          <Line
            key={k}
            type="monotone"
            dataKey={k}
            stroke={colors[k]}
            dot={false}
            strokeWidth={1}
            isAnimationActive={false}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
