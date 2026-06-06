from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.aqi import aqi_category
from src.config import HORIZONS, ALERT_THRESHOLD, TABULAR_FEATURES
from src.feature_pipeline import run_pipeline
from src.model_registry import predict_multi_horizon, load_best_model

st.set_page_config(page_title="Karachi AQI Predictor", page_icon="🌫️", layout="wide")

# AQI category -> (gradient start, gradient end, text color)
CAT_THEME = {
    "Good": ("#43c463", "#2ba84a", "#ffffff"),
    "Moderate": ("#f7d038", "#eab308", "#1f2937"),
    "Unhealthy for Sensitive Groups": ("#fb923c", "#f97316", "#ffffff"),
    "Unhealthy": ("#ef4444", "#dc2626", "#ffffff"),
    "Very Unhealthy": ("#a855f7", "#7e22ce", "#ffffff"),
    "Hazardous": ("#9f1239", "#7e0023", "#ffffff"),
}

HEALTH_MSG = {
    "Good": "Air quality is satisfactory and poses little or no risk.",
    "Moderate": "Air quality is acceptable. Unusually sensitive people should limit prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Sensitive groups may experience health effects. Limit prolonged outdoor exertion.",
    "Unhealthy": "Everyone may begin to experience health effects. Reduce time outdoors.",
    "Very Unhealthy": "Health alert: everyone may experience more serious effects. Avoid outdoor activity.",
    "Hazardous": "Health warning of emergency conditions. Stay indoors and keep windows closed.",
}

POLLUTANTS = [
    ("PM2.5", "pm2_5", "µg/m³"),
    ("PM10", "pm10", "µg/m³"),
    ("Ozone", "ozone", "µg/m³"),
    ("NO₂", "nitrogen_dioxide", "µg/m³"),
    ("SO₂", "sulphur_dioxide", "µg/m³"),
    ("CO", "carbon_monoxide", "µg/m³"),
]


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


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"], .stMarkdown, button, input, select, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%); }
    .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1080px; }

    .app-head { display:flex; align-items:center; justify-content:space-between; margin-bottom:1.2rem; }
    .app-title { font-size:1.9rem; font-weight:800; color:#0f172a; letter-spacing:-0.02em; margin:0; }
    .app-sub { font-size:0.85rem; color:#64748b; margin-top:2px; }
    .pill { background:#fff; border:1px solid #e2e8f0; border-radius:999px; padding:6px 14px;
            font-size:0.8rem; color:#475569; font-weight:600; box-shadow:0 1px 2px rgba(15,23,42,0.04); }

    .hero { border-radius:22px; padding:30px 34px; color:#fff; margin-bottom:18px;
            box-shadow:0 12px 30px rgba(15,23,42,0.12); }
    .hero-grid { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:18px; }
    .hero-aqi { font-size:5.2rem; font-weight:900; line-height:1; letter-spacing:-0.04em; }
    .hero-unit { font-size:0.9rem; font-weight:700; opacity:0.85; text-transform:uppercase; letter-spacing:0.08em; }
    .hero-cat { font-size:1.5rem; font-weight:800; margin-top:4px; }
    .hero-msg { font-size:0.92rem; opacity:0.92; max-width:420px; line-height:1.45; margin-top:6px; }
    .hero-badge { background:rgba(255,255,255,0.18); border:1px solid rgba(255,255,255,0.35);
                  border-radius:14px; padding:14px 20px; text-align:center; backdrop-filter:blur(4px); }

    .section-title { font-size:1.05rem; font-weight:800; color:#0f172a; margin:22px 0 12px; letter-spacing:-0.01em; }

    .card { background:#fff; border:1px solid #eef2f7; border-radius:16px; padding:16px 18px;
            box-shadow:0 4px 14px rgba(15,23,42,0.05); height:100%; }
    .p-label { font-size:0.78rem; font-weight:700; color:#64748b; text-transform:uppercase; letter-spacing:0.05em; }
    .p-value { font-size:1.7rem; font-weight:800; color:#0f172a; line-height:1.1; margin-top:6px; }
    .p-unit { font-size:0.78rem; color:#94a3b8; font-weight:600; }

    .fc-card { border-radius:16px; padding:18px; color:#fff; box-shadow:0 6px 18px rgba(15,23,42,0.10); }
    .fc-h { font-size:0.82rem; font-weight:700; opacity:0.9; text-transform:uppercase; letter-spacing:0.06em; }
    .fc-when { font-size:0.78rem; opacity:0.85; margin-bottom:8px; }
    .fc-aqi { font-size:2.6rem; font-weight:900; line-height:1; letter-spacing:-0.03em; }
    .fc-cat { font-size:0.9rem; font-weight:700; margin-top:6px; }

    .alert { border-radius:14px; padding:14px 18px; background:#fef2f2; border:1px solid #fecaca;
             color:#991b1b; font-weight:600; font-size:0.92rem; margin-bottom:14px; }
    .foot { color:#94a3b8; font-size:0.78rem; text-align:center; margin-top:28px; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Fetching latest air-quality data..."):
    df = get_current_df()
    latest = df.iloc[-1]
    aqi_now = int(latest["aqi"])
    cat_now = aqi_category(aqi_now)
    forecast = get_forecast(df)

updated = pd.Timestamp(latest["datetime"]).strftime("%a %d %b, %H:%M")

st.markdown(
    f"""
    <div class="app-head">
      <div>
        <p class="app-title">🌫️ Karachi Air Quality</p>
        <div class="app-sub">3-day US-EPA AQI forecast · machine-learning models on Open-Meteo data</div>
      </div>
      <div class="pill">📍 Karachi, Pakistan · updated {updated}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

max_fc = max(v["predicted_aqi"] for v in forecast.values())
if max_fc > ALERT_THRESHOLD or aqi_now > ALERT_THRESHOLD:
    st.markdown(
        f'<div class="alert">⚠️ Unhealthy air expected — AQI forecast to reach '
        f'{max_fc:.0f}. Limit outdoor exposure and consider a mask outdoors.</div>',
        unsafe_allow_html=True,
    )

# Hero
g1, g2, txt = CAT_THEME[cat_now]
st.markdown(
    f"""
    <div class="hero" style="background:linear-gradient(135deg,{g1},{g2});color:{txt};">
      <div class="hero-grid">
        <div>
          <div class="hero-unit">Current US AQI</div>
          <div class="hero-aqi">{aqi_now}</div>
          <div class="hero-cat">{cat_now}</div>
          <div class="hero-msg">{HEALTH_MSG[cat_now]}</div>
        </div>
        <div class="hero-badge">
          <div style="font-size:0.75rem;font-weight:700;opacity:0.85;text-transform:uppercase;letter-spacing:0.06em;">PM2.5</div>
          <div style="font-size:2.2rem;font-weight:900;">{latest['pm2_5']:.0f}</div>
          <div style="font-size:0.75rem;opacity:0.85;">µg/m³</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pollutants
st.markdown('<div class="section-title">Pollutant Breakdown</div>', unsafe_allow_html=True)
cols = st.columns(6)
for col, (label, key, unit) in zip(cols, POLLUTANTS):
    col.markdown(
        f"""
        <div class="card">
          <div class="p-label">{label}</div>
          <div class="p-value">{latest[key]:.1f}</div>
          <div class="p-unit">{unit}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Forecast
st.markdown('<div class="section-title">3-Day Forecast</div>', unsafe_allow_html=True)
fcols = st.columns(3)
for col, (h, v) in zip(fcols, sorted(forecast.items())):
    fg1, fg2, ftxt = CAT_THEME[v["category"]]
    when = pd.Timestamp(v["datetime"]).strftime("%a %d %b · %H:%M")
    col.markdown(
        f"""
        <div class="fc-card" style="background:linear-gradient(135deg,{fg1},{fg2});color:{ftxt};">
          <div class="fc-h">+{h} hours</div>
          <div class="fc-when">{when}</div>
          <div class="fc-aqi">{v['predicted_aqi']:.0f}</div>
          <div class="fc-cat">{v['category']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Historical
st.markdown('<div class="section-title">Historical AQI</div>', unsafe_allow_html=True)
days = st.radio("Window", [7, 14, 30], horizontal=True,
                format_func=lambda d: f"{d} days", label_visibility="collapsed")
hist = get_historical_df(df, days)
area = (
    alt.Chart(hist)
    .mark_area(
        line={"color": "#3b82f6", "strokeWidth": 2},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="#dbeafe", offset=0),
                   alt.GradientStop(color="#3b82f6", offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
    )
    .encode(
        x=alt.X("datetime:T", title=None),
        y=alt.Y("aqi:Q", title="US AQI"),
        tooltip=[alt.Tooltip("datetime:T", title="Time"),
                 alt.Tooltip("aqi:Q", title="AQI", format=".0f")],
    )
    .properties(height=300)
    .configure_view(strokeWidth=0)
    .configure_axis(grid=True, gridColor="#eef2f7", labelColor="#64748b", titleColor="#64748b")
)
st.altair_chart(area, use_container_width=True)

# Model performance (tucked away)
with st.expander("Model performance (RMSE / MAE / R²)"):
    metrics = get_metrics()
    mtabs = st.tabs([f"+{h}h" for h in HORIZONS])
    for tab, h in zip(mtabs, HORIZONS):
        rows = [
            {"Model": name, "RMSE": round(m["rmse"], 2), "MAE": round(m["mae"], 2), "R²": round(m["r2"], 3)}
            for name, m in metrics[h].items()
        ]
        tab.dataframe(pd.DataFrame(rows).sort_values("RMSE"),
                      hide_index=True, use_container_width=True)

with st.expander("What drives the forecast (SHAP feature importance)"):
    shap_h = st.selectbox("Horizon", HORIZONS, format_func=lambda h: f"+{h}h")
    with st.spinner("Computing SHAP values..."):
        imp = get_shap(shap_h, df).head(15)
    st.bar_chart(imp, horizontal=True, color="#8b5cf6", height=400)

st.markdown(
    '<div class="foot">Data: Open-Meteo (CAMS air quality + weather) · '
    'Forecast horizons +24h / +48h / +72h · US-EPA AQI scale</div>',
    unsafe_allow_html=True,
)
