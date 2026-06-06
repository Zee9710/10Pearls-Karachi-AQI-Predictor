import { useState, useEffect } from 'react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from 'recharts'

interface ShapData {
  features: string[]
  importance: number[]
  model_type: string
  horizon: number
}

export default function ShapChart() {
  const [horizon, setHorizon] = useState(24)
  const [data, setData] = useState<ShapData | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    setLoading(true)
    fetch(`/api/shap?horizon=${horizon}`)
      .then(r => r.json())
      .then(setData)
      .finally(() => setLoading(false))
  }, [horizon])

  const chartData = data
    ? data.features
        .map((f, i) => ({ feature: f, importance: data.importance[i] }))
        .sort((a, b) => b.importance - a.importance)
        .slice(0, 15)
    : []

  return (
    <section className="mb-6 bg-white rounded-xl shadow p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold text-gray-700">SHAP Feature Importance</h2>
        <select
          value={horizon}
          onChange={e => setHorizon(Number(e.target.value))}
          className="border rounded px-2 py-1 text-sm"
        >
          {[24, 48, 72].map(h => <option key={h} value={h}>+{h}h</option>)}
        </select>
      </div>
      {data && (
        <p className="text-xs text-gray-400 mb-3">
          Model type: <strong>{data.model_type}</strong> — showing mean |SHAP value| per feature
        </p>
      )}
      {loading ? (
        <p className="text-gray-400 text-center py-8">Computing SHAP values...</p>
      ) : (
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={chartData} layout="vertical" margin={{ left: 120, right: 20 }}>
            <XAxis type="number" />
            <YAxis type="category" dataKey="feature" width={115} tick={{ fontSize: 11 }} />
            <Tooltip formatter={(v) => Number(v).toFixed(4)} />
            <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
              {chartData.map((_, i) => (
                <Cell key={i} fill={`hsl(${220 - i * 12}, 70%, 55%)`} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      )}
    </section>
  )
}
