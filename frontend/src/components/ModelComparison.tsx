import { useState, useEffect } from 'react'

interface ModelMetrics {
  rmse: number
  mae: number
  r2: number
}

type MetricsData = Record<string, Record<string, ModelMetrics>>

const MODEL_NAMES = ['ridge', 'random_forest', 'xgboost', 'lightgbm', 'lstm']
const HORIZONS = ['24', '48', '72']

export default function ModelComparison() {
  const [data, setData] = useState<MetricsData | null>(null)

  useEffect(() => {
    fetch('/api/metrics').then(r => r.json()).then(setData)
  }, [])

  if (!data) return null

  return (
    <section className="mb-6 bg-white rounded-xl shadow p-5">
      <h2 className="text-xl font-semibold text-gray-700 mb-4">Model Performance</h2>
      <p className="text-xs text-gray-400 mb-3">RMSE / MAE / R² for all 5 models × 3 forecast horizons</p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="bg-gray-50">
              <th className="px-3 py-2 text-left border border-gray-200">Model</th>
              {HORIZONS.map(h => (
                <th key={h} colSpan={3} className="px-3 py-2 text-center border border-gray-200">
                  +{h}h
                </th>
              ))}
            </tr>
            <tr className="bg-gray-50 text-gray-500">
              <th className="px-3 py-1 border border-gray-200"></th>
              {HORIZONS.flatMap(h => ['RMSE', 'MAE', 'R²'].map(m => (
                <th key={`${h}-${m}`} className="px-2 py-1 border border-gray-200 font-normal">{m}</th>
              )))}
            </tr>
          </thead>
          <tbody>
            {MODEL_NAMES.map(model => (
              <tr key={model} className="hover:bg-blue-50">
                <td className="px-3 py-2 font-medium border border-gray-200 capitalize">
                  {model.replace('_', ' ')}
                </td>
                {HORIZONS.flatMap(h => {
                  const m = data[h]?.[model]
                  return [
                    <td key={`${h}-rmse`} className="px-2 py-2 text-center border border-gray-200">
                      {m ? m.rmse.toFixed(2) : '—'}
                    </td>,
                    <td key={`${h}-mae`} className="px-2 py-2 text-center border border-gray-200">
                      {m ? m.mae.toFixed(2) : '—'}
                    </td>,
                    <td key={`${h}-r2`} className="px-2 py-2 text-center border border-gray-200">
                      {m ? m.r2.toFixed(3) : '—'}
                    </td>,
                  ]
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
