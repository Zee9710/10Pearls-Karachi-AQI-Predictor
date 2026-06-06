from __future__ import annotations

import requests
import pandas as pd
from datetime import date, timedelta

from src.config import LATITUDE, LONGITUDE, TIMEZONE


_AQ_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_AQ_VARS = "pm2_5,pm10,ozone,nitrogen_dioxide,sulphur_dioxide,carbon_monoxide"
_WEATHER_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,surface_pressure"


def _parse_hourly(data: dict, columns: list[str]) -> pd.DataFrame:
    hourly = data["hourly"]
    df = pd.DataFrame({"datetime": hourly["time"]})
    for col in columns:
        df[col] = hourly[col]
    df["datetime"] = pd.to_datetime(df["datetime"]).dt.tz_localize(TIMEZONE)
    df = df.ffill().bfill()
    return df


def fetch_air_quality(start_date: str, end_date: str) -> pd.DataFrame:
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": _AQ_VARS,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
    }
    resp = requests.get(_AQ_URL, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_hourly(
        resp.json(),
        ["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"],
    )


def fetch_weather(start_date: str, end_date: str) -> pd.DataFrame:
    cutoff = (date.today() - timedelta(days=7)).isoformat()
    url = _ARCHIVE_URL if end_date <= cutoff else _FORECAST_URL
    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "hourly": _WEATHER_VARS,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": TIMEZONE,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return _parse_hourly(
        resp.json(),
        ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "surface_pressure"],
    )
