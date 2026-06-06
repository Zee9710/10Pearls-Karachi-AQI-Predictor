from __future__ import annotations

import requests
import numpy as np
import pandas as pd
from datetime import date, datetime, timedelta

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


from src.aqi import compute_aqi


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("datetime").reset_index(drop=True)

    df["aqi"] = df.apply(
        lambda r: compute_aqi(
            r["pm2_5"], r["pm10"], r["ozone"],
            r["nitrogen_dioxide"], r["sulphur_dioxide"], r["carbon_monoxide"],
        ),
        axis=1,
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["datetime"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["datetime"].dt.hour / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["datetime"].dt.dayofweek / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["datetime"].dt.dayofweek / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["datetime"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["datetime"].dt.month / 12)
    df["is_weekend"] = (df["datetime"].dt.dayofweek >= 5).astype(int)

    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_24h"] = df["aqi"].shift(24)
    df["aqi_lag_48h"] = df["aqi"].shift(48)
    df["aqi_lag_72h"] = df["aqi"].shift(72)

    df["aqi_roll24_mean"] = df["aqi"].shift(1).rolling(24).mean()
    df["aqi_roll24_std"] = df["aqi"].shift(1).rolling(24).std()
    df["aqi_roll72_mean"] = df["aqi"].shift(1).rolling(72).mean()
    df["aqi_roll72_max"] = df["aqi"].shift(1).rolling(72).max()

    df["aqi_change_24h"] = df["aqi"].diff(24)
    df["temp_humidity"] = df["temperature_2m"] * df["relative_humidity_2m"]
    df["wind_pollution"] = df["wind_speed_10m"] * df["pm2_5"]

    df = df.dropna().reset_index(drop=True)
    return df


def run_pipeline(start_date: str, end_date: str) -> pd.DataFrame:
    aq_df = fetch_air_quality(start_date, end_date)
    wx_df = fetch_weather(start_date, end_date)
    df = aq_df.merge(wx_df, on="datetime", how="inner")
    return compute_features(df)


import hopsworks
from src.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
)


def _get_feature_group():
    project = hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )
    fs = project.get_feature_store()
    fg = fs.get_or_create_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
        primary_key=["datetime"],
        event_time="datetime",
        online_enabled=False,
    )
    return fg


def write_to_feature_store(df: pd.DataFrame) -> None:
    fg = _get_feature_group()
    fg.insert(df, write_options={"upsert": True})


def run_hourly_update() -> None:
    end = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow() - timedelta(hours=3)).date().isoformat()
    df = run_pipeline(start, end)
    write_to_feature_store(df)


if __name__ == "__main__":
    run_hourly_update()
