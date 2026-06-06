import os
from datetime import date, timedelta

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.aqi import aqi_category
from src.config import HORIZONS, ALERT_THRESHOLD, TABULAR_FEATURES
from src.feature_pipeline import run_pipeline
from src.feature_store import read_recent
from src.model_registry import predict_multi_horizon, load_best_model

st.set_page_config(page_title="Karachi Air Quality", layout="wide")

# Make Supabase credentials available whether running locally (.env) or on
# Streamlit Cloud (st.secrets), so the app can read the Feature Store.
load_dotenv()
try:
    for _k in ("SUPABASE_URL", "SUPABASE_KEY"):
        if _k in st.secrets:
            os.environ[_k] = st.secrets[_k]
except Exception:
    pass

WHO_PM25_ANNUAL = 5.0

# AQI palette: category -> (vivid color, light tint, dark text)
CAT = {
    "Good": ("#22c55e", "#eafaf0", "#15803d"),
    "Moderate": ("#eab308", "#fef9e7", "#a16207"),
    "Unhealthy for Sensitive Groups": ("#f97316", "#fff1e6", "#c2410c"),
    "Unhealthy": ("#ef4444", "#fdeaea", "#b91c1c"),
    "Very Unhealthy": ("#a855f7", "#f5ecfd", "#7e22ce"),
    "Hazardous": ("#9f1239", "#fbe9ee", "#881337"),
}

AQI_RANGES = [
    ("Good", "0–50"),
    ("Moderate", "51–100"),
    ("Unhealthy for Sensitive Groups", "101–150"),
    ("Unhealthy", "151–200"),
    ("Very Unhealthy", "201–300"),
    ("Hazardous", "301+"),
]

HEALTH_MSG = {
    "Good": "Air quality is satisfactory and poses little or no risk.",
    "Moderate": "Acceptable air quality. Unusually sensitive people should limit prolonged outdoor exertion.",
    "Unhealthy for Sensitive Groups": "Sensitive groups may experience health effects. Limit prolonged outdoor exertion.",
    "Unhealthy": "Everyone may begin to experience health effects. Reduce time outdoors.",
    "Very Unhealthy": "Health alert: everyone may experience more serious effects. Avoid outdoor activity.",
    "Hazardous": "Health warning of emergency conditions. Stay indoors and keep windows closed.",
}

POLLUTANTS = [
    ("PM2.5", "pm2_5", "#f43f5e"),
    ("PM10", "pm10", "#f97316"),
    ("Ozone", "ozone", "#0ea5e9"),
    ("NO2", "nitrogen_dioxide", "#8b5cf6"),
    ("SO2", "sulphur_dioxide", "#14b8a6"),
    ("CO", "carbon_monoxide", "#64748b"),
]

NICE_MODEL = {
    "ridge": "Ridge Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "lstm": "LSTM",
}

FEATURE_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "ozone": "Ozone",
    "nitrogen_dioxide": "NO2", "sulphur_dioxide": "SO2", "carbon_monoxide": "CO",
    "temperature_2m": "Temperature", "relative_humidity_2m": "Humidity",
    "wind_speed_10m": "Wind speed", "precipitation": "Precipitation", "surface_pressure": "Pressure",
    "hour_sin": "Hour (sin)", "hour_cos": "Hour (cos)",
    "dow_sin": "Day-of-week (sin)", "dow_cos": "Day-of-week (cos)",
    "month_sin": "Month (sin)", "month_cos": "Month (cos)", "is_weekend": "Weekend",
    "aqi_lag_1h": "AQI 1h ago", "aqi_lag_24h": "AQI 24h ago",
    "aqi_lag_48h": "AQI 48h ago", "aqi_lag_72h": "AQI 72h ago",
    "aqi_roll24_mean": "AQI 24h avg", "aqi_roll24_std": "AQI 24h volatility",
    "aqi_roll72_mean": "AQI 72h avg", "aqi_roll72_max": "AQI 72h max",
    "aqi_change_24h": "AQI 24h change", "temp_humidity": "Temp x Humidity",
    "wind_pollution": "Wind x PM2.5",
}

# White-stroke line icons for the gradient KPI cards
_W = 'width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"'
ICON_PM = f'<svg {_W}><path d="M2 12h20M2 6h14M2 18h10"/></svg>'
ICON_TEMP = f'<svg {_W}><path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4 4 0 1 0 5 0z"/></svg>'
ICON_WIND = f'<svg {_W}><path d="M9.59 4.59A2 2 0 1 1 11 8H2m10.59 11.41A2 2 0 1 0 14 16H2m15.73-8.27A2.5 2.5 0 1 1 19.5 12H2"/></svg>'
ICON_DROP = f'<svg {_W}><path d="M12 2.69l5.66 5.66a8 8 0 1 1-11.31 0z"/></svg>'

KPI_GRAD = {
    "pm": "linear-gradient(135deg,#f43f5e,#fb7185)",
    "temp": "linear-gradient(135deg,#f97316,#fbbf24)",
    "wind": "linear-gradient(135deg,#06b6d4,#22d3ee)",
    "hum": "linear-gradient(135deg,#3b82f6,#60a5fa)",
}


@st.cache_data(ttl=3600)
def load_data() -> tuple[pd.DataFrame, str]:
    """Load features from the Feature Store (per spec); fall back to a live
    Open-Meteo fetch so the app still works without Supabase credentials."""
    store_err = None
    try:
        df = read_recent(n=1000)
        if not df.empty:
            return df, "Feature Store"
        store_err = "feature store returned no rows"
    except Exception as e:
        store_err = str(e)
    try:
        end = date.today()
        start = end - timedelta(days=34)
        return run_pipeline(start.isoformat(), end.isoformat()), "Live Open-Meteo"
    except Exception as live_err:
        raise RuntimeError(
            f"Could not load data. Feature Store: {store_err}. "
            f"Live fallback: {live_err}. "
            "Set SUPABASE_URL and SUPABASE_KEY in the Streamlit Cloud app "
            "secrets so the app reads from the Feature Store."
        ) from live_err


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
def get_champions() -> dict:
    return {h: load_best_model(h)[2] for h in HORIZONS}


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


def gauge_pct(aqi: float) -> float:
    return min(aqi / 300.0, 1.0) * 100.0


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');
    html, body, [class*="css"], .stMarkdown, button, input, select, textarea {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    #MainMenu, footer, header { visibility: hidden; }
    .stApp { background:
        radial-gradient(1200px 600px at 12% -8%, #e0ecff 0%, rgba(224,236,255,0) 55%),
        radial-gradient(1000px 500px at 105% 0%, #f3e8ff 0%, rgba(243,232,255,0) 50%),
        #f6f8fc; }
    .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1180px; }

    section[data-testid="stSidebar"] { background:#0f172a; }
    section[data-testid="stSidebar"] * { color:#e2e8f0; }
    .sb-brand { font-size:1.15rem; font-weight:900; color:#fff; letter-spacing:-0.02em; }
    .sb-brand span { color:#60a5fa; }
    .sb-sub { font-size:0.76rem; color:#94a3b8; margin:2px 0 18px; }
    .sb-box { background:#1e293b; border:1px solid #334155; border-radius:14px; padding:13px 15px; margin-bottom:14px; }
    .sb-k { font-size:0.68rem; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8; font-weight:700; }
    .sb-v { font-size:0.95rem; font-weight:700; color:#f8fafc; margin-top:3px; }
    .sb-live { display:inline-flex; align-items:center; gap:7px; font-size:0.8rem; font-weight:700; color:#86efac; }
    .sb-live .dot { width:8px; height:8px; border-radius:50%; background:#22c55e;
                    box-shadow:0 0 0 3px rgba(34,197,94,0.25); }
    .sb-legend-title { font-size:0.7rem; text-transform:uppercase; letter-spacing:0.06em; color:#94a3b8; font-weight:700; margin:6px 0 8px; }
    .lg-row { display:flex; align-items:center; gap:9px; margin:6px 0; font-size:0.82rem; color:#cbd5e1; }
    .lg-dot { width:11px; height:11px; border-radius:3px; flex:none; }
    .lg-rng { margin-left:auto; color:#94a3b8; font-size:0.76rem; font-weight:600; }

    .hero-title { font-size:1.9rem; font-weight:900; color:#0f172a; letter-spacing:-0.03em; margin:0; }
    .hero-sub { font-size:0.88rem; color:#64748b; margin:3px 0 18px; }

    .card { background:#fff; border:1px solid #e8edf5; border-radius:22px;
            box-shadow:0 10px 30px rgba(15,23,42,0.06); }
    .gcard { padding:24px; text-align:center; height:100%; }
    .gauge { width:188px; height:188px; border-radius:50%; margin:6px auto 0;
             display:flex; align-items:center; justify-content:center; }
    .gauge-inner { width:150px; height:150px; border-radius:50%; background:#fff;
                   display:flex; flex-direction:column; align-items:center; justify-content:center; }
    .gauge-num { font-size:3rem; font-weight:900; line-height:1; }
    .gauge-cap { font-size:0.7rem; font-weight:800; letter-spacing:0.1em; color:#94a3b8; text-transform:uppercase; margin-top:4px; }
    .pill { display:inline-flex; align-items:center; gap:8px; padding:8px 16px; border-radius:999px;
            font-size:0.95rem; font-weight:800; margin-top:16px; }
    .pill .pdot { width:9px; height:9px; border-radius:50%; }
    .gmeta { font-size:0.82rem; color:#64748b; margin-top:12px; font-weight:600; }

    .kpi-grid { display:grid; grid-template-columns:1fr 1fr; gap:16px; height:100%; }
    .kpi { border-radius:20px; padding:18px 20px; color:#fff; position:relative; overflow:hidden;
           box-shadow:0 10px 24px rgba(15,23,42,0.10); display:flex; flex-direction:column; justify-content:space-between; }
    .kpi .ic { opacity:0.9; }
    .kpi .kv { font-size:2rem; font-weight:900; line-height:1; margin-top:14px; }
    .kpi .kk { font-size:0.78rem; font-weight:700; opacity:0.92; margin-top:5px; text-transform:uppercase; letter-spacing:0.04em; }

    .sect { font-size:1.15rem; font-weight:900; color:#0f172a; margin:26px 0 14px; letter-spacing:-0.02em; }

    .fc { border-radius:20px; padding:20px 16px; text-align:center; background:#fff;
          border:1px solid #e8edf5; box-shadow:0 8px 22px rgba(15,23,42,0.05); }
    .fc-day { font-size:0.95rem; font-weight:900; color:#0f172a; }
    .fc-when { font-size:0.74rem; color:#94a3b8; margin-bottom:14px; }
    .fc-ring { width:92px; height:92px; border-radius:50%; margin:0 auto;
               display:flex; align-items:center; justify-content:center; }
    .fc-ring-in { width:72px; height:72px; border-radius:50%; background:#fff; display:flex;
                  align-items:center; justify-content:center; font-size:1.7rem; font-weight:900; color:#0f172a; }
    .fc-cat { font-size:0.82rem; font-weight:800; margin-top:13px; }

    .pcard { background:#fff; border:1px solid #e8edf5; border-radius:18px; padding:16px 18px;
             box-shadow:0 6px 18px rgba(15,23,42,0.05); }
    .pcard .pl { font-size:0.78rem; font-weight:800; color:#0f172a; }
    .pcard .pv { font-size:1.5rem; font-weight:900; color:#0f172a; margin-top:6px; }
    .pcard .pu { font-size:0.72rem; color:#cbd5e1; font-weight:700; }
    .pbar { height:6px; border-radius:999px; background:#eef2f7; margin-top:12px; overflow:hidden; }
    .pbar span { display:block; height:100%; border-radius:999px; }

    .champ { display:flex; align-items:center; gap:10px; background:linear-gradient(135deg,#eff6ff,#eef2ff);
             border:1px solid #bfdbfe; border-radius:14px; padding:12px 16px; margin:4px 0 16px;
             font-size:0.92rem; color:#1e3a8a; font-weight:800; }
    .champ .tag { background:#3b82f6; color:#fff; font-size:0.66rem; font-weight:900; letter-spacing:0.06em;
                  padding:4px 10px; border-radius:999px; text-transform:uppercase; }

    .alert { border-radius:16px; padding:15px 20px; background:linear-gradient(135deg,#fff7ed,#fef2f2);
             border:1px solid #fed7aa; border-left:6px solid #f97316; color:#9a3412;
             font-weight:700; font-size:0.92rem; margin-bottom:18px;
             box-shadow:0 8px 22px rgba(249,115,22,0.10); }
    .foot { color:#94a3b8; font-size:0.78rem; text-align:center; margin-top:34px; }

    .stTabs [data-baseweb="tab-list"] { gap:6px; }
    .stTabs [data-baseweb="tab"] { background:#fff; border:1px solid #e8edf5; border-radius:12px 12px 0 0;
                                   padding:8px 18px; font-weight:700; }
    .stTabs [aria-selected="true"] { background:#3b82f6; color:#fff !important; border-color:#3b82f6; }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Fetching latest air-quality data..."):
    try:
        df, source = load_data()
    except Exception as e:
        st.error(str(e))
        st.stop()
    latest = df.iloc[-1]
    aqi_now = int(latest["aqi"])
    cat_now = aqi_category(aqi_now)
    forecast = get_forecast(df)

updated = pd.Timestamp(latest["datetime"]).strftime("%a %d %b, %H:%M")
ring, tint, txt = CAT[cat_now]
who_mult = latest["pm2_5"] / WHO_PM25_ANNUAL
pct = gauge_pct(aqi_now)

# ---------------- Sidebar ----------------
legend_html = "".join(
    f'<div class="lg-row"><span class="lg-dot" style="background:{CAT[c][0]};"></span>{c}'
    f'<span class="lg-rng">{rng}</span></div>'
    for c, rng in AQI_RANGES
)
st.sidebar.markdown(
    f"""
    <div class="sb-brand">Air<span>Cast</span></div>
    <div class="sb-sub">Karachi AQI intelligence dashboard</div>
    <div class="sb-box">
      <div class="sb-k">Location</div>
      <div class="sb-v">Karachi, Pakistan</div>
      <div class="sb-k" style="margin-top:10px;">Coordinates</div>
      <div class="sb-v">24.86° N, 67.00° E</div>
    </div>
    <div class="sb-box">
      <div class="sb-k">Last updated</div>
      <div class="sb-v">{updated}</div>
      <div style="margin-top:9px;"><span class="sb-live"><span class="dot"></span>LIVE · {source}</span></div>
    </div>
    <div class="sb-legend-title">AQI scale</div>
    {legend_html}
    """,
    unsafe_allow_html=True,
)

# ---------------- Header ----------------
st.markdown(
    '<p class="hero-title">Karachi Air Quality</p>'
    '<div class="hero-sub">Real-time conditions and a 3-day US-EPA AQI forecast, '
    'powered by machine-learning models on Open-Meteo data</div>',
    unsafe_allow_html=True,
)

max_fc = max(v["predicted_aqi"] for v in forecast.values())
if max_fc > ALERT_THRESHOLD or aqi_now > ALERT_THRESHOLD:
    st.markdown(
        f'<div class="alert">⚠ Unhealthy air expected — AQI forecast to reach '
        f'{max_fc:.0f}. Limit outdoor exposure and consider a mask outdoors.</div>',
        unsafe_allow_html=True,
    )

# ---------------- Hero row: gauge + KPI grid ----------------
left, right = st.columns([1.04, 1.96], gap="medium")

left.markdown(
    f"""
    <div class="card gcard">
      <div class="gauge" style="background:conic-gradient({ring} {pct}%, #eef2f7 {pct}% 100%);">
        <div class="gauge-inner">
          <div class="gauge-num" style="color:{ring};">{aqi_now}</div>
          <div class="gauge-cap">US AQI</div>
        </div>
      </div>
      <div class="pill" style="background:{tint};color:{txt};">
        <span class="pdot" style="background:{ring};"></span>{cat_now}
      </div>
      <div class="gmeta">PM2.5 {latest['pm2_5']:.1f} µg/m³ · {who_mult:.1f}× WHO guideline</div>
    </div>
    """,
    unsafe_allow_html=True,
)

right.markdown(
    f"""
    <div class="kpi-grid">
      <div class="kpi" style="background:{KPI_GRAD['pm']};">
        <div class="ic">{ICON_PM}</div>
        <div><div class="kv">{latest['pm2_5']:.0f}</div><div class="kk">PM2.5 µg/m³</div></div>
      </div>
      <div class="kpi" style="background:{KPI_GRAD['temp']};">
        <div class="ic">{ICON_TEMP}</div>
        <div><div class="kv">{latest['temperature_2m']:.0f}°</div><div class="kk">Temperature</div></div>
      </div>
      <div class="kpi" style="background:{KPI_GRAD['wind']};">
        <div class="ic">{ICON_WIND}</div>
        <div><div class="kv">{latest['wind_speed_10m']:.0f}</div><div class="kk">Wind km/h</div></div>
      </div>
      <div class="kpi" style="background:{KPI_GRAD['hum']};">
        <div class="ic">{ICON_DROP}</div>
        <div><div class="kv">{latest['relative_humidity_2m']:.0f}%</div><div class="kk">Humidity</div></div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f'<div class="card" style="padding:16px 22px;margin-top:16px;'
            f'font-size:0.92rem;color:#475569;font-weight:600;">{HEALTH_MSG[cat_now]}</div>',
            unsafe_allow_html=True)

# ---------------- Forecast ----------------
st.markdown('<div class="sect">3-Day Forecast</div>', unsafe_allow_html=True)
fcols = st.columns(3, gap="medium")
for col, (h, v) in zip(fcols, sorted(forecast.items())):
    fring, ftint, ftxt = CAT[v["category"]]
    fpct = gauge_pct(v["predicted_aqi"])
    when = pd.Timestamp(v["datetime"]).strftime("%a %d %b · %H:%M")
    col.markdown(
        f"""
        <div class="fc">
          <div class="fc-day">+{h} hours</div>
          <div class="fc-when">{when}</div>
          <div class="fc-ring" style="background:conic-gradient({fring} {fpct}%, #eef2f7 {fpct}% 100%);">
            <div class="fc-ring-in">{v['predicted_aqi']:.0f}</div>
          </div>
          <div class="fc-cat" style="color:{ftxt};">{v['category']}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------- Interactive trend + forecast ----------------
st.markdown('<div class="sect">AQI Trend & Forecast</div>', unsafe_allow_html=True)
days = st.radio("Window", [7, 14, 30], horizontal=True, index=1,
                format_func=lambda d: f"Last {d} days", label_visibility="collapsed")
hist = get_historical_df(df, days)[["datetime", "aqi"]].copy()

fc_rows = pd.DataFrame([
    {"datetime": pd.Timestamp(v["datetime"]), "aqi": v["predicted_aqi"]}
    for v in forecast.values()
])
last_point = hist.iloc[[-1]][["datetime", "aqi"]]
fc_line = pd.concat([last_point, fc_rows], ignore_index=True)

hover = alt.selection_point(fields=["datetime"], nearest=True, on="mouseover", empty=False)

area = (
    alt.Chart(hist).mark_area(
        line={"color": "#3b82f6", "strokeWidth": 2.5},
        color=alt.Gradient(
            gradient="linear",
            stops=[alt.GradientStop(color="#dbeafe", offset=0),
                   alt.GradientStop(color="#3b82f6", offset=1)],
            x1=1, x2=1, y1=1, y2=0,
        ),
    ).encode(
        x=alt.X("datetime:T", title=None),
        y=alt.Y("aqi:Q", title="US AQI"),
    )
)
threshold = (
    alt.Chart(pd.DataFrame({"y": [ALERT_THRESHOLD]}))
    .mark_rule(color="#ef4444", strokeDash=[6, 4], strokeWidth=1.5)
    .encode(y="y:Q")
)
fc_path = (
    alt.Chart(fc_line).mark_line(color="#a855f7", strokeWidth=2.5, strokeDash=[5, 4],
                                 point=alt.OverlayMarkDef(color="#a855f7", size=70))
    .encode(x="datetime:T", y="aqi:Q",
            tooltip=[alt.Tooltip("datetime:T", title="Time"),
                     alt.Tooltip("aqi:Q", title="Forecast AQI", format=".0f")])
)
points = (
    alt.Chart(hist).mark_circle(size=60, color="#3b82f6")
    .encode(x="datetime:T", y="aqi:Q",
            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
            tooltip=[alt.Tooltip("datetime:T", title="Time"),
                     alt.Tooltip("aqi:Q", title="AQI", format=".0f")])
    .add_params(hover)
)
chart = (
    (area + threshold + fc_path + points)
    .properties(height=320)
    .configure_view(strokeWidth=0)
    .configure_axis(grid=True, gridColor="#eef2f7", labelColor="#64748b", titleColor="#94a3b8")
)
st.altair_chart(chart, use_container_width=True)
st.caption("Solid blue = observed history · dashed purple = model forecast · red dashed line = AQI 150 (Unhealthy).")

# ---------------- Tabbed analytics ----------------
st.markdown('<div class="sect">Details</div>', unsafe_allow_html=True)
tab_p, tab_m, tab_s = st.tabs(["Pollutants", "Model Performance", "Feature Importance"])

with tab_p:
    pmax = {"pm2_5": 250, "pm10": 430, "ozone": 240, "nitrogen_dioxide": 200,
            "sulphur_dioxide": 350, "carbon_monoxide": 15400}
    pcols = st.columns(3, gap="medium")
    for i, (label, key, color) in enumerate(POLLUTANTS):
        val = latest[key]
        fillpct = min(val / pmax.get(key, 300) * 100, 100)
        pcols[i % 3].markdown(
            f"""
            <div class="pcard" style="margin-bottom:16px;">
              <div class="pl">{label}</div>
              <div class="pv">{val:.1f}</div>
              <div class="pu">µg/m³</div>
              <div class="pbar"><span style="width:{fillpct:.0f}%;background:{color};"></span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with tab_m:
    metrics = get_metrics()
    champions = get_champions()
    mtabs = st.tabs([f"+{h}h" for h in HORIZONS])
    for mt, h in zip(mtabs, HORIZONS):
        with mt:
            champ = champions[h]
            cm = metrics[h][champ]
            st.markdown(
                f'<div class="champ"><span class="tag">Champion</span>'
                f'{NICE_MODEL.get(champ, champ)} &nbsp;·&nbsp; '
                f'RMSE {cm["rmse"]:.2f} &nbsp;·&nbsp; MAE {cm["mae"]:.2f} &nbsp;·&nbsp; '
                f'R² {cm["r2"]:.3f}</div>',
                unsafe_allow_html=True,
            )
            mdf = pd.DataFrame([
                {"Model": NICE_MODEL.get(n, n), "RMSE": m["rmse"],
                 "MAE": m["mae"], "R²": m["r2"], "is_champ": n == champ}
                for n, m in metrics[h].items()
            ]).sort_values("RMSE")
            bars = (
                alt.Chart(mdf).mark_bar(cornerRadiusEnd=6, height=24)
                .encode(
                    x=alt.X("RMSE:Q", title="RMSE (lower is better)"),
                    y=alt.Y("Model:N", sort="-x", title=None),
                    color=alt.condition(alt.datum.is_champ,
                                        alt.value("#3b82f6"), alt.value("#cbd5e1")),
                    tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("RMSE:Q", format=".2f"),
                             alt.Tooltip("MAE:Q", format=".2f"), alt.Tooltip("R²:Q", format=".3f")],
                )
                .properties(height=max(130, 36 * len(mdf)))
                .configure_view(strokeWidth=0)
                .configure_axis(grid=True, gridColor="#eef2f7", labelColor="#64748b", titleColor="#94a3b8")
            )
            st.altair_chart(bars, use_container_width=True)

with tab_s:
    shap_h = st.selectbox("Forecast horizon", HORIZONS, format_func=lambda h: f"+{h} hours")
    with st.spinner("Computing SHAP values..."):
        imp = get_shap(shap_h, df).head(12)
    sdf = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(i, i) for i in imp.index],
        "Importance": imp.values,
    })
    shap_chart = (
        alt.Chart(sdf).mark_bar(cornerRadiusEnd=6, height=22)
        .encode(
            x=alt.X("Importance:Q", title="Mean |SHAP value| — impact on predicted AQI"),
            y=alt.Y("Feature:N", sort="-x", title=None),
            color=alt.Color("Importance:Q", scale=alt.Scale(scheme="bluepurple"), legend=None),
            tooltip=[alt.Tooltip("Feature:N"), alt.Tooltip("Importance:Q", format=".3f")],
        )
        .properties(height=420)
        .configure_view(strokeWidth=0)
        .configure_axis(grid=True, gridColor="#eef2f7", labelColor="#64748b", titleColor="#94a3b8")
    )
    st.altair_chart(shap_chart, use_container_width=True)
    st.caption("Larger bars = features the model leans on most when forecasting AQI.")

st.markdown(
    '<div class="foot">Data: Open-Meteo (CAMS air quality + weather) · '
    'Forecast horizons +24h / +48h / +72h · US-EPA AQI scale</div>',
    unsafe_allow_html=True,
)
