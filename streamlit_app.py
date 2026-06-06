from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

from src.aqi import aqi_category
from src.config import HORIZONS, ALERT_THRESHOLD, TABULAR_FEATURES
from src.feature_pipeline import run_pipeline
from src.model_registry import predict_multi_horizon, load_best_model

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

CATEGORY_COLORS = {
    "Good": "#00e400",
    "Moderate": "#ffff00",
    "Unhealthy for Sensitive Groups": "#ff7e00",
    "Unhealthy": "#ff0000",
    "Very Unhealthy": "#8f3f97",
    "Hazardous": "#7e0023",
}


@st.cache_data(ttl=3600)
def get_current_df() -> pd.DataFrame:
    end = date.today()
    start = end - timedelta(days=34)
    return run_pipeline(start.isoformat(), end.isoformat())


def get_historical_df(df: pd.DataFrame, days: int) -> pd.DataFrame:
    cutoff = pd.Timestamp.now(tz=df["datetime"].dt.tz) - pd.Timedelta(days=days)
    return df[df["datetime"] >= cutoff].reset_index(drop=True)


@st.cache_data(ttl=3600)
def get_forecast(_df: pd.DataFrame) -> dict:
    out = predict_multi_horizon(_df)
    return {
        h: {"predicted_aqi": v["predicted_aqi"], "category": v["category"],
            "datetime": v["datetime"].isoformat()}
        for h, v in out.items()
    }


@st.cache_data(ttl=3600)
def get_metrics() -> dict:
    return {h: load_best_model(h)[3] for h in HORIZONS}


@st.cache_data(ttl=3600)
def get_shap(horizon: int, _df: pd.DataFrame):
    import shap
    model, scaler, model_type, _ = load_best_model(horizon)
    X = _df[TABULAR_FEATURES].values.astype(np.float64)
    X_scaled = scaler.transform(X)
    sample = X_scaled[-min(100, len(X_scaled)):]
    if model_type in ("random_forest", "xgboost", "lightgbm"):
        sv = shap.TreeExplainer(model).shap_values(sample)
    else:
        sv = shap.LinearExplainer(model, sample).shap_values(sample)
    imp = np.abs(sv).mean(axis=0)
    return pd.Series(imp, index=TABULAR_FEATURES).sort_values(ascending=False)


st.title("🌫️ Karachi AQI Predictor")
st.caption("3-day air-quality forecast for Karachi · models trained on Open-Meteo data · US-EPA AQI")

with st.spinner("Fetching latest air-quality data..."):
    df = get_current_df()
    latest = df.iloc[-1]
    aqi_now = int(latest["aqi"])
    cat_now = aqi_category(aqi_now)
    forecast = get_forecast(df)

max_fc = max(v["predicted_aqi"] for v in forecast.values())
if max_fc > ALERT_THRESHOLD or aqi_now > ALERT_THRESHOLD:
    st.error(f"⚠️ Hazardous air quality: AQI is forecast to reach {max_fc:.0f}. Limit outdoor exposure.")

# Current conditions
st.subheader("Current Conditions")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("AQI", aqi_now, cat_now)
c2.metric("PM2.5", f"{latest['pm2_5']:.1f}")
c3.metric("PM10", f"{latest['pm10']:.1f}")
c4.metric("Ozone", f"{latest['ozone']:.1f}")
c5.metric("NO₂", f"{latest['nitrogen_dioxide']:.1f}")

# Forecast
st.subheader("3-Day Forecast")
fc_df = pd.DataFrame([
    {"Horizon": f"+{h}h", "Predicted AQI": v["predicted_aqi"], "Category": v["category"]}
    for h, v in sorted(forecast.items())
])
fcol1, fcol2 = st.columns([2, 1])
fcol1.bar_chart(fc_df.set_index("Horizon")["Predicted AQI"], color="#3b82f6")
fcol2.dataframe(fc_df, hide_index=True, use_container_width=True)

# Historical
st.subheader("Historical AQI")
days = st.radio("Window", [7, 14, 30], horizontal=True, format_func=lambda d: f"{d} days")
hist = get_historical_df(df, days)
st.line_chart(hist.set_index("datetime")["aqi"], color="#3b82f6", height=260)

# Model comparison
st.subheader("Model Comparison (RMSE / MAE / R²)")
metrics = get_metrics()
mtab1, mtab2, mtab3 = st.tabs([f"+{h}h" for h in HORIZONS])
for tab, h in zip([mtab1, mtab2, mtab3], HORIZONS):
    rows = [
        {"Model": name, "RMSE": round(m["rmse"], 2), "MAE": round(m["mae"], 2), "R²": round(m["r2"], 3)}
        for name, m in metrics[h].items()
    ]
    tab.dataframe(pd.DataFrame(rows).sort_values("RMSE"), hide_index=True, use_container_width=True)

# Explainability
st.subheader("Feature Importance (SHAP)")
shap_h = st.selectbox("Horizon", HORIZONS, format_func=lambda h: f"+{h}h")
with st.spinner("Computing SHAP values..."):
    imp = get_shap(shap_h, df).head(15)
st.bar_chart(imp, horizontal=True, color="#8b5cf6", height=400)

st.caption("Data: Open-Meteo (CAMS air quality + weather). Updated hourly. Forecast horizons: +24h / +48h / +72h.")
