import os
from dotenv import load_dotenv

load_dotenv()

LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"
CITY = "Karachi"

HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "karachi_aqi")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

FEATURE_GROUP_NAME = "karachi_aqi_features"
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = "aqi_training_view"
FEATURE_VIEW_VERSION = 1

HORIZONS = [24, 48, 72]
ALERT_THRESHOLD = 150

LSTM_SEQ_LEN = 24
LSTM_RAW_FEATURES = [
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "aqi",
]

TABULAR_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend",
    "aqi_lag_1h", "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
    "aqi_roll24_mean", "aqi_roll24_std", "aqi_roll72_mean", "aqi_roll72_max",
    "aqi_change_24h", "temp_humidity", "wind_pollution",
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "surface_pressure",
    "pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide",
]
