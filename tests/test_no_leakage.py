import pandas as pd
import numpy as np
from src.feature_pipeline import compute_features


def _make_raw_df(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "datetime": dates,
        "pm2_5": rng.uniform(5, 150, n),
        "pm10": rng.uniform(10, 200, n),
        "ozone": rng.uniform(20, 100, n),
        "nitrogen_dioxide": rng.uniform(5, 50, n),
        "sulphur_dioxide": rng.uniform(2, 30, n),
        "carbon_monoxide": rng.uniform(200, 2000, n),
        "temperature_2m": rng.uniform(20, 45, n),
        "relative_humidity_2m": rng.uniform(30, 80, n),
        "wind_speed_10m": rng.uniform(0, 20, n),
        "precipitation": np.zeros(n),
        "surface_pressure": rng.uniform(990, 1010, n),
    })


def test_lag_1h_is_strictly_past():
    raw = _make_raw_df(200)
    df = compute_features(raw)
    df = df.reset_index(drop=True)
    assert not (df["aqi"] == df["aqi_lag_1h"]).all()


def test_rolling_uses_shift_not_current():
    raw = _make_raw_df(200)
    df = compute_features(raw)
    assert not (df["aqi"] == df["aqi_roll24_mean"]).all()


def test_aqi_change_24h_uses_past():
    raw = _make_raw_df(200)
    df = compute_features(raw)
    assert (df["aqi_change_24h"] != 0).any()


def test_lag_72h_present():
    raw = _make_raw_df(200)
    df = compute_features(raw)
    assert "aqi_lag_72h" in df.columns
    assert not df["aqi_lag_72h"].isnull().any()
