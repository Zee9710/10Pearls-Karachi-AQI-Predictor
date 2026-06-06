# Karachi AQI Predictor

End-to-end, 100% serverless machine learning pipeline that forecasts the Air
Quality Index (AQI) for Karachi, Pakistan over the next three days
(+24h / +48h / +72h).

- **Live dashboard:** https://10pearls-karachi-aqi-predictor-v8ygbuwud8du7m3mun22cr.streamlit.app
- **Full write-up:** see [REPORT.md](REPORT.md)

## Stack

- **Data**: Open-Meteo API (free, no key required) — pollutants + weather + historical archive
- **Feature store**: Supabase (Postgres + PostgREST), accessed over plain HTTPS
- **Models**: Ridge, Random Forest, XGBoost, LightGBM, LSTM — 5 models × 3 horizons; champions saved to `models/` and versioned in git
- **Model registry**: `models/` committed to git
- **Explainability**: SHAP (feature importance in the dashboard)
- **Dashboard**: Streamlit (deployed on Streamlit Community Cloud)
- **CI/CD**: GitHub Actions — feature pipeline hourly, training pipeline daily
- **Alternative frontend** (optional): a Flask + React app under `api/` and `frontend/` exposing the same models with SHAP + LIME

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Zee9710/10Pearls-Karachi-AQI-Predictor.git
cd 10Pearls-Karachi-AQI-Predictor
pip install -r requirements-train.txt   # base requirements + TensorFlow (for the LSTM)
```

For the dashboard only (no model training), `requirements.txt` is sufficient.

### 2. Configure the feature store

Create a Supabase project, run `supabase_schema.sql` in its SQL editor to
create the `aqi_features` table, then add credentials to `.env`:

```bash
SUPABASE_URL=https://<your-project>.supabase.co
SUPABASE_KEY=<your-service-key>
```

### 3. Backfill ~1 year of historical data

```bash
python -m src.backfill
```

### 4. Train all 5 models

```bash
python -m src.training_pipeline
```

Champions per horizon are written to `models/aqi_best_{24,48,72}h/`.

### 5. Run the dashboard

```bash
streamlit run streamlit_app.py
```

## GitHub Actions

Add these repository secrets:
- `SUPABASE_URL`
- `SUPABASE_KEY`

Workflows:
- `hourly-feature.yml` — runs `src/feature_pipeline.py` every hour to update features
- `daily-train.yml` — runs `src/training_pipeline.py` daily at 03:00 UTC and commits refreshed models

## Streamlit Cloud deployment

The dashboard is deployed on Streamlit Community Cloud. Two settings matter:
- **Python version**: 3.12 (pinned wheels are unavailable for 3.13).
- **Secrets**: add `SUPABASE_URL` and `SUPABASE_KEY` (TOML format) so the app
  reads from the feature store. Without them it falls back to a live
  Open-Meteo fetch.

## Project Structure

```
10Pearls-Karachi-AQI-Predictor/
├── src/
│   ├── config.py            # Constants: lat/lon, horizons, feature lists
│   ├── aqi.py               # US-EPA AQI breakpoint formula
│   ├── feature_pipeline.py  # Fetch -> compute features -> upsert to Supabase
│   ├── backfill.py          # ~1-year historical seed
│   ├── feature_store.py     # Supabase read/upsert over PostgREST
│   ├── training_pipeline.py # 5 models x 3 horizons, champion selection
│   └── model_registry.py    # Load champions, serve predictions
├── streamlit_app.py         # Streamlit dashboard (primary, deployed)
├── api/                     # Optional Flask API (SHAP + LIME)
├── frontend/                # Optional React + TypeScript dashboard
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── models/                  # Model registry (champions, committed)
├── tests/                   # pytest suite (EPA formula, features, leakage, API)
├── supabase_schema.sql      # Feature-store table definition
└── .github/workflows/       # CI/CD (hourly feature, daily training)
```

## Models

| Horizon | Champion | RMSE | MAE | R² |
|---|---|---|---|---|
| +24h | Ridge | 21.88 | 15.53 | 0.308 |
| +48h | Ridge | 24.59 | 18.07 | 0.128 |
| +72h | Ridge | 25.47 | 18.99 | 0.065 |

AQI on the 0–500 US-EPA scale; metrics on a chronological 80/20 hold-out.
See [REPORT.md](REPORT.md) for the full model comparison and analysis.

## Tests

```bash
pytest
```

Covers the EPA AQI formula, feature engineering, train/test leakage guards,
and the API layer.
