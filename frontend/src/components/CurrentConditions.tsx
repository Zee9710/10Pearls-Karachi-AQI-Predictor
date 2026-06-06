import type { CurrentData } from '../types'
import { AQI_COLORS } from '../types'

interface Props {
  data: CurrentData
}

function MetricCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white rounded-xl shadow p-4 flex flex-col gap-1">
      <span className="text-xs text-gray-500 uppercase tracking-wide">{label}</span>
      <span className="text-2xl font-bold text-gray-800">{value}</span>
    </div>
  )
}

export default function CurrentConditions({ data }: Props) {
  const color = AQI_COLORS[data.category] ?? '#ccc'
  return (
    <section className="mb-6">
      <h2 className="text-xl font-semibold text-gray-700 mb-3">Current Conditions</h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="AQI" value={String(data.aqi)} />
        <MetricCard label="PM2.5 µg/m³" value={String(data.pm2_5)} />
        <MetricCard label="PM10 µg/m³" value={String(data.pm10)} />
        <MetricCard label="O₃ µg/m³" value={String(data.ozone)} />
      </div>
      <div
        className="mt-3 inline-block px-4 py-1 rounded-full text-sm font-semibold text-gray-800"
        style={{ background: color }}
      >
        {data.category}
      </div>
      <p className="text-xs text-gray-400 mt-1">As of {new Date(data.datetime).toLocaleString()}</p>
    </section>
  )
}
