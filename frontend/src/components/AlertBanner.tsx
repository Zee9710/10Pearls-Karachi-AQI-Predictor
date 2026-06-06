import type { ForecastData } from '../types'

interface Props {
  aqi: number
  forecast: ForecastData
}

export default function AlertBanner({ aqi, forecast }: Props) {
  const worstHorizon = Object.entries(forecast).find(
    ([, v]) => v.is_hazardous
  )
  if (!worstHorizon) return null

  return (
    <div className="mb-4 p-4 bg-red-100 border-l-4 border-red-600 rounded flex items-start gap-3">
      <span className="text-2xl">🚨</span>
      <div>
        <p className="font-bold text-red-800 text-lg">Hazardous Air Quality Alert</p>
        <p className="text-red-700">
          Forecast AQI reaching <strong>{aqi.toFixed(0)}</strong> ({worstHorizon[1].category}).
          Limit outdoor activity and wear an N95 mask.
        </p>
      </div>
    </div>
  )
}
