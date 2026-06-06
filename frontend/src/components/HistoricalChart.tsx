import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer } from 'recharts'
import type { HistoricalPoint } from '../types'

export default function HistoricalChart() {
  const [days, setDays] = useState(7)
  const [data, setData] = useState<HistoricalPoint[]>([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetch(`/api/historical?days=${days}`)
      .then(r => r.json())
      .then((d: HistoricalPoint[]) => {
        const step = Math.ceil(d.length / 200)
        setData(d.filter((_, i) => i % step === 0))
      })
      .finally(() => setLoading(false))
  }, [days])

  return (
    <section className="mb-6 bg-white rounded-xl shadow p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-700">Historical AQI</h2>
        <div className="flex gap-2">
          {[7, 14, 30].map(d => (
            <button
              key={d}
              onClick={() => setDays(d)}
              className={`px-3 py-1 rounded text-sm ${days === d ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-700'}`}
            >
              {d}d
            </button>
          ))}
        </div>
      </div>
      {loading ? (
        <p className="text-gray-400 text-center py-10">Loading...</p>
      ) : (
        <ResponsiveContainer width="100%" height={240}>
          <LineChart data={data}>
            <XAxis
              dataKey="datetime"
              tickFormatter={v => new Date(v).toLocaleDateString()}
              interval="preserveStartEnd"
            />
            <YAxis domain={[0, 'auto']} label={{ value: 'AQI', angle: -90, position: 'insideLeft' }} />
            <Tooltip
              labelFormatter={v => new Date(v).toLocaleString()}
              formatter={(val) => [Number(val).toFixed(1), 'AQI']}
            />
            <ReferenceLine y={150} stroke="#ff7e00" strokeDasharray="4 4" />
            <Line type="monotone" dataKey="aqi" stroke="#3b82f6" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      )}
    </section>
  )
}
