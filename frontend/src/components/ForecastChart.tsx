import { BarChart, Bar, XAxis, YAxis, Tooltip, ReferenceLine, Cell, ResponsiveContainer } from 'recharts'
import type { ForecastData } from '../types'
import { AQI_COLORS } from '../types'

interface Props {
  data: ForecastData
}

export default function ForecastChart({ data }: Props) {
  const chartData = Object.entries(data).map(([h, v]) => ({
    horizon: `+${h}h`,
    aqi: v.predicted_aqi,
    category: v.category,
    datetime: new Date(v.datetime).toLocaleString(),
  }))

  return (
    <section className="mb-6 bg-white rounded-xl shadow p-5">
      <h2 className="text-xl font-semibold text-gray-700 mb-4">3-Day Forecast</h2>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={chartData} margin={{ top: 20, right: 20, bottom: 0, left: 0 }}>
          <XAxis dataKey="horizon" />
          <YAxis domain={[0, 500]} label={{ value: 'AQI', angle: -90, position: 'insideLeft' }} />
          <Tooltip
            formatter={(val: number) => [val.toFixed(0), '']}
          />
          <ReferenceLine y={150} stroke="#ff7e00" strokeDasharray="4 4" label="Unhealthy" />
          <Bar dataKey="aqi" radius={[4, 4, 0, 0]}>
            {chartData.map((entry, i) => (
              <Cell key={i} fill={AQI_COLORS[entry.category] ?? '#888'} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </section>
  )
}
