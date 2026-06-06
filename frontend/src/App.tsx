import { useEffect, useState } from 'react'
import AlertBanner from './components/AlertBanner'
import CurrentConditions from './components/CurrentConditions'
import ForecastChart from './components/ForecastChart'
import HistoricalChart from './components/HistoricalChart'
import type { CurrentData, ForecastData } from './types'

export default function App() {
  const [current, setCurrent] = useState<CurrentData | null>(null)
  const [forecast, setForecast] = useState<ForecastData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    Promise.all([
      fetch('/api/current').then(r => r.json()),
      fetch('/api/forecast').then(r => r.json()),
    ])
      .then(([c, f]) => { setCurrent(c); setForecast(f) })
      .catch(() => setError('Could not load data. Is the Flask API running on port 5000?'))
      .finally(() => setLoading(false))
  }, [])

  if (loading) {
    return (
      <div className="flex justify-center items-center h-screen text-gray-500 text-xl">
        Loading AQI data...
      </div>
    )
  }
  if (error) {
    return (
      <div className="flex justify-center items-center h-screen text-red-500 text-center px-4">
        {error}
      </div>
    )
  }

  const maxAqi = forecast
    ? Math.max(...Object.values(forecast).map(f => f.predicted_aqi))
    : 0

  return (
    <div className="min-h-screen bg-gray-50 p-4 md:p-8 max-w-5xl mx-auto">
      <header className="mb-6">
        <h1 className="text-3xl font-bold text-gray-800">Karachi AQI Predictor</h1>
        <p className="text-sm text-gray-500">Powered by Open-Meteo · Hopsworks · 5 ML Models</p>
      </header>

      {forecast && maxAqi > 150 && <AlertBanner aqi={maxAqi} forecast={forecast} />}
      {current && <CurrentConditions data={current} />}
      {forecast && <ForecastChart data={forecast} />}
      <HistoricalChart />
    </div>
  )
}
