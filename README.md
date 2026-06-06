---
title: Karachi AQI Predictor
emoji: 🌫️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 5000
pinned: false
---

# Karachi AQI Predictor

End-to-end machine learning pipeline for hourly AQI forecasting in Karachi, Pakistan.

## Stack

- **Data**: Open-Meteo API (free, no key required)
- **Models**: Ridge, Random Forest, XGBoost, LightGBM, LSTM — 5 models × 3 horizons (24h/48h/72h); champions saved to `models/` and committed to git
- **Explainability**: SHAP (global) + LIME (per-prediction)
- **API**: Flask + Flask-CORS (single-origin; serves the built React app)
- **Frontend**: React 19 + TypeScript + Vite + Recharts + Tailwind CSS
- **CI/CD**: GitHub Actions (daily training auto-commits refreshed models)

## Setup

### 1. Clone and install

```bash
git clone <your-repo-url>
cd karachi-aqi-predictor
pip install -r requirements.txt
```

### 2. Configure secrets

```bash
cp .env.example .env
# Edit .env and add your Hopsworks API key
```

Get a free API key at [app.hopsworks.ai](https://app.hopsworks.ai).

### 3. Backfill 1 year of historical data

```bash
python -m src.backfill
```

### 4. Train all 5 models

```bash
python -m src.training_pipeline
```

### 5. Run the Flask API

```bash
python -m api.app
```

### 6. Run the React frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

## GitHub Actions

Add these secrets to your GitHub repository:
- `HOPSWORKS_API_KEY`
- `HOPSWORKS_PROJECT` (default: `karachi_aqi`)

Workflows:
- `hourly-feature.yml` — runs `feature_pipeline.py` every hour
- `daily-train.yml` — runs `training_pipeline.py` at 03:00 UTC daily

## Project Structure

```
karachi-aqi-predictor/
├── src/
│   ├── config.py            # Constants: lat/lon, Hopsworks, feature lists
│   ├── aqi.py               # US-EPA AQI breakpoint formula
│   ├── feature_pipeline.py  # Fetch → compute features → write to Hopsworks
│   ├── backfill.py          # 1-year historical seed
│   ├── training_pipeline.py # 5 models × 3 horizons
│   └── model_registry.py    # Load models, serve predictions
├── api/
│   ├── app.py               # Flask app factory
│   └── routes.py            # REST endpoints
├── frontend/                # React 18 + TypeScript dashboard
├── notebooks/
│   └── eda.ipynb            # Exploratory data analysis
├── tests/                   # pytest suite
└── .github/workflows/       # CI/CD
```

## Key improvements over reference repo

| Reference issue | This project |
|---|---|
| 1–5 classification | Real EPA AQI regression (0–500) with RMSE/MAE/R² |
| Only 3 models | 5 models (Ridge, RF, XGB, LGBM, LSTM) |
| Faked 3-day forecast | Genuine +24/+48/+72h with independent models |
| MongoDB | Hopsworks feature store with versioning |
| SHAP only | SHAP (global) + LIME (local/per-prediction) |
| No alerts | Hazardous AQI banner at AQI > 150 |
| Destructive backfill | Append-only upsert writes |
| No tests | pytest: EPA formula, features, leakage, API |
