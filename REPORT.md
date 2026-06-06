# Pearls AQI Predictor — Project Report

A 100% serverless, end-to-end machine learning system that forecasts the Air
Quality Index (AQI) for Karachi, Pakistan over the next three days
(+24h, +48h, +72h). The system fetches data, engineers features, trains and
registers models, and serves predictions through an interactive dashboard —
all on free tiers with no manual intervention.

- **Live dashboard:** https://10pearls-karachi-aqi-predictor-v8ygbuwud8du7m3mun22cr.streamlit.app
- **Source:** https://github.com/Zee9710/10Pearls-Karachi-AQI-Predictor
- **City:** Karachi (24.8607° N, 67.0011° E), timezone `Asia/Karachi`

---

## 1. Architecture

```
Open-Meteo API ──► Feature Pipeline ──► Feature Store (Supabase)
   (raw data)        (src/feature_         (aqi_features table)
                      pipeline.py)               │
                                                 ├──► Training Pipeline ──► Model Registry
                                                 │     (src/training_         (models/ in git)
                                                 │      pipeline.py)              │
                                                 └──► Streamlit Dashboard ◄───────┘
                                                       (streamlit_app.py)

   GitHub Actions: feature pipeline hourly · training pipeline daily
```

The design follows the classic feature-store / model-registry MLOps pattern.
Every stage is decoupled: the feature pipeline only writes features, the
training pipeline only reads features and writes models, and the dashboard only
reads. This keeps each component independently testable and replaceable.

### Serverless stack

| Concern | Choice | Why |
|---|---|---|
| Raw data | Open-Meteo API | Free, no API key, provides both air-quality and weather, plus a historical archive endpoint |
| Feature store | Supabase (Postgres + PostgREST) | Free tier, SQL-queryable, accessed over plain HTTPS |
| Model registry | `models/` committed to git | Zero-cost, versioned, deploys with the app |
| Orchestration | GitHub Actions | Free cron scheduling, no server to run |
| Dashboard | Streamlit Community Cloud | Free hosting, direct public URL |

---

## 2. Feature Pipeline (`src/feature_pipeline.py`)

**1. Fetch raw data.** Pulls hourly pollutant data (PM2.5, PM10, ozone, NO₂,
SO₂, CO) from Open-Meteo's air-quality endpoint and weather data
(temperature, humidity, wind speed, precipitation, surface pressure) from the
forecast/archive endpoints. A retry helper (`_get_json`) backs off on HTTP 429
and 5xx responses and honors `Retry-After`, which matters on shared cloud IPs.

**2. Compute the AQI target.** The US-EPA AQI (0–500) is computed from the raw
pollutant concentrations in `src/aqi.py` using the standard breakpoint
piecewise-linear formula, taking the maximum sub-index across pollutants.

**3. Engineer features.** 29 model inputs are derived:

- **Time (cyclical):** `hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`,
  `is_weekend` — sine/cosine encoding so 23:00 and 00:00 are adjacent.
- **Lags:** `aqi_lag_1h`, `aqi_lag_24h`, `aqi_lag_48h`, `aqi_lag_72h`.
- **Rolling stats:** 24h and 72h rolling mean/std/max (shifted to avoid
  leakage).
- **Change rate:** `aqi_change_24h` — the day-over-day AQI delta required by
  the assignment.
- **Interactions:** `temp_humidity`, `wind_pollution`.
- **Raw signals:** all six pollutants plus five weather variables.

**4. Store.** Features are upserted into the Supabase `aqi_features` table,
keyed on `datetime` so re-runs are idempotent (no duplicate rows).

---

## 3. Backfill (`src/backfill.py`)

The same feature script is run over a one-year range of past dates to generate
training data. This populated the feature store with roughly one year of
hourly rows (~8,900 records from 2025-06-01 onward). The backfill uses the
Open-Meteo archive endpoint for dates older than seven days and the forecast
endpoint for recent dates.

---

## 4. Training Pipeline (`src/training_pipeline.py`)

**1. Fetch from the feature store.** Reads the full feature history from
Supabase (not from a live API) — the training stage never touches the raw API.

**2. Train and evaluate.** For each horizon, the target is the AQI shifted
forward (`aqi.shift(-horizon)`), giving direct multi-horizon forecasting. The
data is split 80/20 chronologically (no shuffling — this respects time order
and prevents look-ahead leakage). Five model families are trained and compared:

| Family | Type |
|---|---|
| Ridge Regression (`RidgeCV`) | Statistical / linear |
| Random Forest | Bagged trees |
| XGBoost | Gradient boosting |
| LightGBM | Gradient boosting |
| LSTM (TensorFlow/Keras) | Deep learning (sequence model) |

This satisfies the "variety from statistical modelling to deep learning"
guideline. Each model is scored on **RMSE, MAE, and R²**; the lowest-RMSE
model per horizon is promoted to champion.

**3. Register.** The champion model (plus its scaler and metrics) is written to
`models/aqi_best_{24,48,72}h/` and committed to git by the daily CI job.

### Results

Champion per horizon (lowest RMSE). AQI is on the 0–500 EPA scale.

| Horizon | Champion | RMSE | MAE | R² |
|---|---|---|---|---|
| +24h | Ridge | 21.88 | 15.53 | 0.308 |
| +48h | Ridge | 24.59 | 18.07 | 0.128 |
| +72h | Ridge | 25.47 | 18.99 | 0.065 |

Full model comparison (RMSE):

| Model | +24h | +48h | +72h |
|---|---|---|---|
| **Ridge** | **21.88** | **24.59** | **25.47** |
| Random Forest | 22.95 | 28.86 | 28.05 |
| XGBoost | 23.74 | 28.30 | 27.10 |
| LightGBM | 24.24 | 28.64 | 27.49 |
| LSTM | 29.81 | 29.90 | 29.60 |

**Observations.**
- Ridge wins at every horizon. With ~8.9k hourly rows and strongly
  autocorrelated lag features, the linear model generalizes better than the
  tree ensembles and the LSTM, which overfit (negative R² on the held-out
  tail for the longer horizons).
- Accuracy decays with horizon, as expected: +24h is the most predictable
  (R² ≈ 0.31), while +72h approaches the persistence baseline (R² ≈ 0.07).
- The LSTM underperforms here — a known pattern when the dataset is modest and
  the linear lag signal is dominant. It is retained in the comparison to
  demonstrate the statistical-to-deep-learning spectrum and would benefit from
  more data as the feature store grows daily.

---

## 5. CI/CD Automation (`.github/workflows/`)

| Workflow | Schedule | Action |
|---|---|---|
| `hourly-feature.yml` | `0 * * * *` (hourly) | Fetch the latest window, compute features, upsert to the feature store |
| `daily-train.yml` | `0 3 * * *` (daily) | Read the feature store, retrain all models, commit updated champions |

Both run on GitHub Actions (free), authenticate to Supabase via repository
secrets (`SUPABASE_URL`, `SUPABASE_KEY`), and require no server. The training
job installs `requirements-train.txt` (which adds TensorFlow on top of the base
requirements) and pushes refreshed models back to the repo.

---

## 6. Web App (`streamlit_app.py`)

**Loads model and features from the feature store.** On load the app reads the
most recent feature rows directly from Supabase and loads the champion models
from the registry. If credentials are absent it falls back to a live Open-Meteo
fetch so the demo still works; the active source is shown in the UI.

**Computes predictions and displays them.** The dashboard, styled after IQAir,
shows:

- Current AQI with category ring, health advisory, and WHO PM2.5 multiplier.
- A live status pill indicating the data source and last update time.
- Six pollutant concentration cards.
- Three-day forecast cards (+24h / +48h / +72h) with color-coded categories.
- A historical AQI trend chart (7 / 14 / 30 day windows).
- Model performance section: RMSE comparison across all five models per
  horizon, champion highlighted.
- SHAP feature-importance section explaining which inputs drive each forecast.

**Hazardous-AQI alerts.** When a forecast crosses the alert threshold
(AQI 150, "Unhealthy"), the dashboard surfaces a prominent warning.

---

## 7. Guidelines Coverage

| Guideline | How it is met |
|---|---|
| EDA to identify trends | `notebooks/eda.ipynb` — distributions, correlations, time-of-day and seasonal trends |
| Variety of models (statistical → deep learning) | Ridge → Random Forest → XGBoost → LightGBM → LSTM |
| SHAP / LIME explainability | SHAP feature importance integrated into the dashboard |
| Alerts for hazardous AQI | Threshold alert at AQI 150 in the dashboard |

---

## 8. Deliverables Checklist

1. **End-to-end AQI prediction system** — complete (data → features → models → predictions).
2. **Scalable, automated pipeline** — complete (hourly + daily GitHub Actions, idempotent feature store).
3. **Interactive dashboard** — complete (live + forecasted AQI on Streamlit Cloud).
4. **Detailed report** — this document.

---

## 9. How to Run Locally

```bash
pip install -r requirements-train.txt          # base + TensorFlow

# Configure feature-store credentials
echo 'SUPABASE_URL=...'  >> .env
echo 'SUPABASE_KEY=...'  >> .env

python -m src.backfill            # one-time: populate the feature store
python -m src.feature_pipeline    # hourly update (also run by CI)
python -m src.training_pipeline   # train + register champions
streamlit run streamlit_app.py    # launch the dashboard
```

---

## 10. Limitations and Future Work

- **Forecast skill at long horizons.** +72h R² is low; richer exogenous
  signals (e.g. wind direction, regional transport, traffic) could help.
- **LSTM needs more data.** As the hourly pipeline accumulates history, the
  deep model should be re-benchmarked — it is currently data-starved.
- **Single location.** The pipeline is parameterized by lat/lon in
  `src/config.py` and could be extended to multiple cities.
- **Model registry in git.** Adequate for this scale; a managed registry
  (e.g. MLflow) would be the next step for larger teams or model lineage.
