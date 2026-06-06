import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import pytest
from src.feature_pipeline import fetch_air_quality, fetch_weather


# ---- helpers ----

def _make_aq_response(n=5):
    times = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "hourly": {
            "time": times,
            "pm2_5": [20.0] * n,
            "pm10": [40.0] * n,
            "ozone": [80.0] * n,
            "nitrogen_dioxide": [30.0] * n,
            "sulphur_dioxide": [10.0] * n,
            "carbon_monoxide": [500.0] * n,
        }
    }


def _make_weather_response(n=5):
    times = pd.date_range("2024-01-01", periods=n, freq="h").strftime("%Y-%m-%dT%H:%M").tolist()
    return {
        "hourly": {
            "time": times,
            "temperature_2m": [25.0] * n,
            "relative_humidity_2m": [60.0] * n,
            "wind_speed_10m": [10.0] * n,
            "precipitation": [0.0] * n,
            "surface_pressure": [1013.0] * n,
        }
    }


# ---- fetch_air_quality tests ----

def test_fetch_air_quality_returns_dataframe():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_aq_response()
    with patch("requests.get", return_value=mock_resp):
        df = fetch_air_quality("2024-01-01", "2024-01-05")
    assert isinstance(df, pd.DataFrame)


def test_fetch_air_quality_has_expected_columns():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_aq_response()
    with patch("requests.get", return_value=mock_resp):
        df = fetch_air_quality("2024-01-01", "2024-01-05")
    assert set(["pm2_5", "pm10", "ozone", "nitrogen_dioxide", "sulphur_dioxide", "carbon_monoxide"]).issubset(df.columns)


def test_fetch_air_quality_datetime_is_index_or_column():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_aq_response()
    with patch("requests.get", return_value=mock_resp):
        df = fetch_air_quality("2024-01-01", "2024-01-05")
    assert "datetime" in df.columns or df.index.name == "datetime"


def test_fetch_air_quality_no_nans():
    data = _make_aq_response()
    data["hourly"]["pm2_5"][2] = None
    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    with patch("requests.get", return_value=mock_resp):
        df = fetch_air_quality("2024-01-01", "2024-01-05")
    assert df["pm2_5"].isna().sum() == 0


# ---- fetch_weather tests ----

def test_fetch_weather_returns_dataframe():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_weather_response()
    with patch("requests.get", return_value=mock_resp):
        df = fetch_weather("2024-01-01", "2024-01-05")
    assert isinstance(df, pd.DataFrame)


def test_fetch_weather_has_expected_columns():
    mock_resp = MagicMock()
    mock_resp.json.return_value = _make_weather_response()
    with patch("requests.get", return_value=mock_resp):
        df = fetch_weather("2024-01-01", "2024-01-05")
    expected = {"temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "surface_pressure"}
    assert expected.issubset(df.columns)


def test_fetch_weather_no_nans():
    data = _make_weather_response()
    data["hourly"]["temperature_2m"][1] = None
    mock_resp = MagicMock()
    mock_resp.json.return_value = data
    with patch("requests.get", return_value=mock_resp):
        df = fetch_weather("2024-01-01", "2024-01-05")
    assert df["temperature_2m"].isna().sum() == 0


from src.feature_pipeline import compute_features


def _make_raw_df(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    rng = np.random.default_rng(42)
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


def test_compute_features_returns_dataframe():
    df = compute_features(_make_raw_df(200))
    assert isinstance(df, pd.DataFrame)


def test_compute_features_has_aqi():
    df = compute_features(_make_raw_df(200))
    assert "aqi" in df.columns
    assert (df["aqi"] >= 0).all()


def test_compute_features_no_nan():
    df = compute_features(_make_raw_df(200))
    assert not df.isnull().any().any(), df.isnull().sum()[df.isnull().sum() > 0]


def test_compute_features_expected_columns():
    df = compute_features(_make_raw_df(200))
    expected = [
        "aqi", "hour_sin", "hour_cos", "dow_sin", "dow_cos",
        "month_sin", "month_cos", "is_weekend",
        "aqi_lag_1h", "aqi_lag_24h", "aqi_lag_48h", "aqi_lag_72h",
        "aqi_roll24_mean", "aqi_roll24_std", "aqi_roll72_mean", "aqi_roll72_max",
        "aqi_change_24h", "temp_humidity", "wind_pollution",
    ]
    for col in expected:
        assert col in df.columns, f"Missing: {col}"


def test_compute_features_drops_initial_nan_rows():
    df = compute_features(_make_raw_df(200))
    assert len(df) < 200
    assert len(df) > 0
