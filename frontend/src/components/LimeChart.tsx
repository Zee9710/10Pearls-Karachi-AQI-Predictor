import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ReferenceLine, ResponsiveContainer } from 'recharts'

interface LimeData {
  features: string[]
  weights: number[]
  horizon: number
  model_type: string
}

export default function LimeChart() {
  const [horizon, setHorizon] = useState(24)
  const [data, setData] = useState<LimeData | null>(null)
  const [loading, setLoading] = useState(false)

  function fetchLime() {
    setLoading(true)
    fetch(`/api/lime?horizon=${horizon}`)
      .then(r => r.json())
      .then(setData)
      .finally(() => setLoading(false))
  }

  const chartData = data
    ? data.features
        .map((f, i) => ({ feature: f, weight: data.weights[i] }))
        .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    : []

  return (
    <section className="mb-6 bg-white rounded-xl shadow p-5">
      <div className="flex items-center justify-between mb-4">
        <div>
          <h2 className="text-xl font-semibold text-gray-700">LIME Local Explanation</h2>
          <p className="text-xs text-gray-400">Explains a single prediction with a local linear model</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={horizon}
            onChange={e => setHorizon(Number(e.target.value))}
            className="border rounded px-2 py-1 text-sm"
          >
            {[24, 48, 72].map(h => <option key={h} value={h}>+{h}h</option>)}
          </select>
          <button
            onClick={fetchLime}
            disabled={loading}
            className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Computing...' : 'Explain'}
          </button>
        </div>
      </div>
      {chartData.length > 0 && (
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 140, right: 20 }}>
            <XAxis type="number" label={{ value: 'Contribution to AQI', position: 'insideBottom', offset: -5 }} />
            <YAxis type="category" dataKey="feature" width={135} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => Number(v).toFixed(4)} />
            <ReferenceLine x={0} stroke="#888" />
            <Bar dataKey="weight" radius={[0, 4, 4, 0]}>
              {chartData.map((entry, i) => (
                <Cell key={i} fill={entry.weight < 0 ? '#ef4444' : '#22c55e'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
      {!data && !loading && (
        <p className="text-gray-400 text-center py-6">Click "Explain" to generate a LIME explanation for the latest prediction.</p>
      )}
    </section>
  )
}
