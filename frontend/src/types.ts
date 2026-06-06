export interface CurrentData {
  datetime: string
  aqi: number
  category: string
  pm2_5: number
  pm10: number
  ozone: number
  nitrogen_dioxide: number
  is_hazardous: boolean
}

export interface HorizonForecast {
  predicted_aqi: number
  category: string
  datetime: string
  is_hazardous: boolean
  all_metrics: Record<string, Record<string, number>>
}

export type ForecastData = Record<string, HorizonForecast>

export interface HistoricalPoint {
  datetime: string
  aqi: number
  category: string
}

export const AQI_COLORS: Record<string, string> = {
  Good: '#00e400',
  Moderate: '#ffff00',
  'Unhealthy for Sensitive Groups': '#ff7e00',
  Unhealthy: '#ff0000',
  'Very Unhealthy': '#8f3f97',
  Hazardous: '#7e0023',
}
