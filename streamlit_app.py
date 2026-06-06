from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.aqi import aqi_category
from src.config import HORIZONS, ALERT_THRESHOLD, TABULAR_FEATURES
from src.feature_pipeline import run_pipeline
from src.model_registry import predict_multi_horizon, load_best_model

st.set_page_config(page_title="Karachi Air Quality", layout="wide")

WHO_PM25_ANNUAL = 5.0

# IQAir-style AQI palette: category -> (ring color, light tint, dark text)
CAT = {
    "Good": ("#A8E05F", "#eef9df", "#4d7c0f"),
    "Moderate": ("#FDD64B", "#fef6da", "#a16207"),
    "Unhealthy for Sensitive Groups": ("#FF9B57", "#ffeede", "#c2410c"),
    "Unhealthy": ("#FE6A69", "#ffe1e1", "#b91c1c"),
    "Very Unhealthy": ("#A97ABC", "#f0e6f4", "#7e22ce"),
    "Hazardous": ("#A87383", "#f1e2e7", "#9f1239"),
}

HEALTH_MSG = {
    "Good": "Air quality is satisfactory and poses little or no risk.",
    "Moderate": "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Sensitive groups may experience health effects. Limit prolonged outdoor exertion.",
    "Unhealthy": "Everyone may begin to experience health effects. Reduce time outdoors.",
    "Very Unhealthy": "Health alert: everyone may experience more serious effects. Avoid outdoor activity.",
    "Hazardous": "Health warning of emergency conditions. Stay indoors and keep windows closed.",
}

POLLUTANTS = [
    ("PM2.5", "pm2_5"),
    ("PM10", "pm10"),
    ("Ozone", "ozone"),
    ("NO2", "nitrogen_dioxide"),
    ("SO2", "sulphur_dioxide"),
    ("CO", "carbon_monoxide"),
]

_S = 'width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#64748b" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
ICON_TEMP = f'<svg {_S}><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4 4 0 1 0 5 0z"/></svg>'
ICON_WIND = f'<svg {_S}><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg>'
ICON_DROP = f'<svg {_S}><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>'


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
    .stApp { background: #f4f6f9; }
    .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1040px; }

    .page-title { font-size:1.7rem; font-weight:800; color:#0f172a; letter-spacing:-0.02em; margin:0; }
    .page-sub { font-size:0.85rem; color:#64748b; margin:2px 0 18px; }

    .iq-card { background:#fff; border:1px solid #eaeef3; border-radius:24px; padding:26px 30px;
               box-shadow:0 6px 24px rgba(15,23,42,0.06); }
    .iq-loc { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;
              border-bottom:1px solid #f1f5f9; padding-bottom:14px; margin-bottom:18px; }
    .iq-city { font-size:1.1rem; font-weight:800; color:#0f172a; }
    .iq-live { font-size:0.78rem; font-weight:600; color:#64748b; display:flex; align-items:center; gap:6px; }
    .iq-live .dot { width:8px; height:8px; border-radius:50%; background:#22c55e; display:inline-block;
                    box-shadow:0 0 0 3px rgba(34,197,94,0.18); }

    .iq-main { display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:26px; }
    .iq-aqi-label { font-size:0.75rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.08em; }
    .iq-aqi-row { display:flex; align-items:center; gap:20px; margin-top:8px; }
    .iq-circle { width:108px; height:108px; border-radius:50%; border:9px solid; display:flex;
                 align-items:center; justify-content:center; font-size:2.6rem; font-weight:900; color:#0f172a; }
    .iq-pill { display:inline-flex; align-items:center; gap:8px; padding:7px 14px; border-radius:999px;
               font-size:0.92rem; font-weight:700; }
    .iq-pill .pdot { width:9px; height:9px; border-radius:50%; }
    .iq-pm { font-size:0.9rem; color:#334155; font-weight:600; margin-top:10px; }
    .iq-who { font-size:0.8rem; color:#94a3b8; margin-top:2px; }

    .iq-weather { display:flex; gap:26px; }
    .iq-wx { text-align:center; }
    .iq-wx .v { font-size:1.05rem; font-weight:800; color:#0f172a; margin-top:4px; }
    .iq-wx .k { font-size:0.72rem; color:#94a3b8; font-weight:600; text-transform:uppercase; letter-spacing:0.04em; }

    .iq-msg { margin-top:18px; padding-top:16px; border-top:1px solid #f1f5f9;
              font-size:0.9rem; color:#475569; line-height:1.5; }

    .section { font-size:1.05rem; font-weight:800; color:#0f172a; margin:26px 0 12px; letter-spacing:-0.01em; }

    .card { background:#fff; border:1px solid #eaeef3; border-radius:16px; padding:16px 18px;
            box-shadow:0 3px 12px rgba(15,23,42,0.04); height:100%; }
    .p-label { font-size:0.76rem; font-weight:700; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }
    .p-value { font-size:1.6rem; font-weight:800; color:#0f172a; line-height:1.1; margin-top:6px; }
    .p-unit { font-size:0.74rem; color:#cbd5e1; font-weight:600; }

    .fc { background:#fff; border:1px solid #eaeef3; border-radius:18px; padding:20px 16px; text-align:center;
          box-shadow:0 3px 12px rgba(15,23,42,0.04); }
    .fc-day { font-size:0.82rem; font-weight:800; color:#0f172a; }
    .fc-when { font-size:0.74rem; color:#94a3b8; margin-bottom:14px; }
    .fc-circle { width:76px; height:76px; border-radius:50%; border:7px solid; display:flex; margin:0 auto;
                 align-items:center; justify-content:center; font-size:1.7rem; font-weight:900; color:#0f172a; }
    .fc-cat { font-size:0.82rem; font-weight:700; margin-top:12px; }

    .alert { border-radius:14px; padding:14px 18px; background:#fff7ed; border:1px solid #fed7aa;
             border-left:5px solid #f97316; color:#9a3412; font-weight:600; font-size:0.9rem; margin-bottom:16px; }
    .foot { color:#94a3b8; font-size:0.78rem; text-align:center; margin-top:30px; }
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
ring, tint, txt = CAT[cat_now]
who_mult = latest["pm2_5"] / WHO_PM25_ANNUAL

st.markdown(
    f'<p class="page-title">Karachi Air Quality</p>'
    f'<div class="page-sub">3-day US-EPA AQI forecast · machine-learning models on Open-Meteo data</div>',
    unsafe_allow_html=True,
)

max_fc = max(v["predicted_aqi"] for v in forecast.values())
if max_fc > ALERT_THRESHOLD or aqi_now > ALERT_THRESHOLD:
    st.markdown(
        f'<div class="alert">Unhealthy air expected — AQI forecast to reach '
        f'{max_fc:.0f}. Limit outdoor exposure and consider a mask outdoors.</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f"""
    <div class="iq-card">
      <div class="iq-loc">
        <span class="iq-city">Karachi, Pakistan</span>
        <span class="iq-live"><span class="dot"></span>LIVE · updated {updated}</span>
      </div>
      <div class="iq-main">
        <div>
          <div class="iq-aqi-label">US AQI</div>
          <div class="iq-aqi-row">
            <div class="iq-circle" style="border-color:{ring};">{aqi_now}</div>
            <div>
              <span class="iq-pill" style="background:{tint};color:{txt};">
                <span class="pdot" style="background:{ring};"></span>{cat_now}</span>
              <div class="iq-pm">PM2.5 · {latest['pm2_5']:.1f} µg/m³</div>
              <div class="iq-who">{who_mult:.1f}× the WHO annual guideline</div>
            </div>
          </div>
        </div>
        <div class="iq-weather">
          <div class="iq-wx">{ICON_TEMP}<div class="v">{latest['temperature_2m']:.0f}°C</div><div class="k">Temp</div></div>
          <div class="iq-wx">{ICON_WIND}<div class="v">{latest['wind_speed_10m']:.0f}</div><div class="k">km/h</div></div>
          <div class="iq-wx">{ICON_DROP}<div class="v">{latest['relative_humidity_2m']:.0f}%</div><div class="k">Humidity</div></div>
        </div>
      </div>
      <div class="iq-msg">{HEALTH_MSG[cat_now]}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Pollutants
st.markdown('<div class="section">Pollutant Concentrations</div>', unsafe_allow_html=True)
cols = st.columns(6)
for col, (label, key) in zip(cols, POLLUTANTS):
    col.markdown(
        f"""
        <div class="card">
          <div class="p-label">{label}</div>
          <div class="p-value">{latest[key]:.1f}</div>
          <div class="p-unit">µg/m³</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Forecast
st.markdown('<div class="section">3-Day Forecast</div>', unsafe_allow_html=True)
fcols = st.columns(3)
for col, (h, v) in zip(fcols, sorted(forecast.items())):
    fring, ftint, ftxt = CAT[v["category"]]
    when = pd.Timestamp(v["datetime"]).strftime("%a %d %b · %H:%M")
    col.markdown(
        f"""
        <div class="fc">
          <div class="fc-day">+{h} hours</div>
          <div class="fc-when">{when}</div>
          <div class="fc-circle" style="border-color:{fring};">{v['predicted_aqi']:.0f}</div>
          <div class="fc-cat" style="color:{ftxt};">{v['category']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Historical
st.markdown('<div class="section">Historical AQI</div>', unsafe_allow_html=True)
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
