import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def _fake_current_df():
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "datetime": dates,
        "aqi": rng.integers(50, 200, 100).astype(float),
        "pm2_5": rng.uniform(20, 80, 100),
        "pm10": rng.uniform(40, 120, 100),
        "ozone": rng.uniform(30, 90, 100),
        "nitrogen_dioxide": rng.uniform(10, 50, 100),
        "sulphur_dioxide": rng.uniform(5, 25, 100),
        "carbon_monoxide": rng.uniform(300, 1200, 100),
    })


def _fake_forecasts():
    base = pd.Timestamp("2024-01-05")
    return {
        24: {"predicted_aqi": 80.0, "category": "Moderate",
             "datetime": base + pd.Timedelta(hours=24), "all_metrics": {"ridge": {"rmse": 12.0, "mae": 9.0, "r2": 0.85}}},
        48: {"predicted_aqi": 90.0, "category": "Moderate",
             "datetime": base + pd.Timedelta(hours=48), "all_metrics": {}},
        72: {"predicted_aqi": 160.0, "category": "Unhealthy",
             "datetime": base + pd.Timedelta(hours=72), "all_metrics": {}},
    }


@patch("api.routes._get_current_df")
@patch("api.routes.predict_multi_horizon")
def test_forecast_returns_three_horizons(mock_pred, mock_df, client):
    mock_df.return_value = _fake_current_df()
    mock_pred.return_value = _fake_forecasts()
    resp = client.get("/api/forecast")
    assert resp.status_code == 200
    data = resp.get_json()
    assert set(data.keys()) == {"24", "48", "72"}


@patch("api.routes._get_current_df")
@patch("api.routes.predict_multi_horizon")
def test_forecast_hazardous_flag(mock_pred, mock_df, client):
    mock_df.return_value = _fake_current_df()
    mock_pred.return_value = _fake_forecasts()
    resp = client.get("/api/forecast")
    data = resp.get_json()
    assert data["72"]["is_hazardous"] is True
    assert data["24"]["is_hazardous"] is False


@patch("api.routes._get_current_df")
def test_current_has_required_fields(mock_df, client):
    mock_df.return_value = _fake_current_df()
    resp = client.get("/api/current")
    assert resp.status_code == 200
    data = resp.get_json()
    for field in ["aqi", "category", "pm2_5", "pm10", "datetime", "is_hazardous"]:
        assert field in data


@patch("api.routes.load_historical_features")
def test_historical_returns_list(mock_hist, client):
    mock_hist.return_value = pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
        "aqi": [80.0, 85.0, 90.0, 95.0, 100.0],
    })
    resp = client.get("/api/historical?days=1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) == 5
    assert "datetime" in data[0]
    assert "aqi" in data[0]
    assert "category" in data[0]
