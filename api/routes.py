from flask import Blueprint, jsonify, request
from datetime import datetime, timedelta
import numpy as np

from src.model_registry import (
    predict_multi_horizon, load_historical_features, load_best_model, _get_project,
)
from src.feature_pipeline import run_pipeline
from src.aqi import aqi_category
from src.config import ALERT_THRESHOLD, HORIZONS, TABULAR_FEATURES, LSTM_RAW_FEATURES

bp = Blueprint("api", __name__, url_prefix="/api")


def _get_current_df():
    today = datetime.utcnow().date().isoformat()
    start = (datetime.utcnow() - timedelta(hours=80)).date().isoformat()
    return run_pipeline(start, today)


@bp.route("/current")
def current():
    df = _get_current_df()
    latest = df.iloc[-1]
    aqi_val = int(latest["aqi"])
    return jsonify({
        "datetime": latest["datetime"].isoformat(),
        "aqi": aqi_val,
        "category": aqi_category(aqi_val),
        "pm2_5": round(float(latest["pm2_5"]), 1),
        "pm10": round(float(latest["pm10"]), 1),
        "ozone": round(float(latest["ozone"]), 1),
        "nitrogen_dioxide": round(float(latest["nitrogen_dioxide"]), 1),
        "is_hazardous": aqi_val > ALERT_THRESHOLD,
    })


@bp.route("/forecast")
def forecast():
    df = _get_current_df()
    forecasts = predict_multi_horizon(df)
    result = {}
    for h, v in forecasts.items():
        result[str(h)] = {
            "predicted_aqi": v["predicted_aqi"],
            "category": v["category"],
            "datetime": v["datetime"].isoformat(),
            "is_hazardous": v["predicted_aqi"] > ALERT_THRESHOLD,
            "all_metrics": v.get("all_metrics", {}),
        }
    return jsonify(result)


@bp.route("/historical")
def historical():
    days = request.args.get("days", 7, type=int)
    df = load_historical_features(days=days)
    rows = df[["datetime", "aqi"]].copy()
    rows["category"] = rows["aqi"].apply(lambda x: aqi_category(int(x)))
    rows["datetime"] = rows["datetime"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    rows["aqi"] = rows["aqi"].round(1)
    return jsonify(rows.to_dict(orient="records"))


@bp.route("/metrics")
def metrics():
    project = _get_project()
    result = {}
    for h in HORIZONS:
        _, _, _, all_metrics = load_best_model(project, h)
        result[str(h)] = all_metrics
    return jsonify(result)


@bp.route("/shap")
def shap_endpoint():
    import shap
    horizon = request.args.get("horizon", 24, type=int)
    project = _get_project()
    model, scaler, model_type, _ = load_best_model(project, horizon)

    df = _get_current_df()
    feat_cols = LSTM_RAW_FEATURES if model_type == "lstm" else TABULAR_FEATURES
    X = df[feat_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    sample = shap.sample(X_scaled, min(100, len(X_scaled)))

    if model_type in ("random_forest", "xgboost", "lightgbm"):
        sv = shap.TreeExplainer(model).shap_values(sample)
    elif model_type == "ridge":
        sv = shap.LinearExplainer(model, sample).shap_values(sample)
    else:
        def pred_fn(x):
            return model.predict(
                x.reshape(-1, 1, len(LSTM_RAW_FEATURES)), verbose=0
            ).flatten()
        sv = shap.KernelExplainer(pred_fn, shap.sample(X_scaled, 10)).shap_values(sample[:20])

    mean_abs = np.abs(sv).mean(axis=0).tolist()
    return jsonify({
        "features": feat_cols,
        "importance": mean_abs,
        "model_type": model_type,
        "horizon": horizon,
    })


@bp.route("/lime")
def lime_endpoint():
    from lime import lime_tabular
    horizon = request.args.get("horizon", 24, type=int)
    project = _get_project()
    model, scaler, model_type, _ = load_best_model(project, horizon)

    df = _get_current_df()
    feat_cols = LSTM_RAW_FEATURES if model_type == "lstm" else TABULAR_FEATURES
    X_all = df[feat_cols].values.astype(np.float32)
    X_scaled = scaler.transform(X_all)

    if model_type == "lstm":
        def predict_fn(x):
            return model.predict(x.reshape(-1, 1, len(LSTM_RAW_FEATURES)), verbose=0).flatten()
    else:
        def predict_fn(x):
            return model.predict(x)

    lime_exp = lime_tabular.LimeTabularExplainer(
        X_scaled, feature_names=feat_cols, mode="regression", random_state=42,
    )
    explanation = lime_exp.explain_instance(X_scaled[-1], predict_fn, num_features=10)
    items = explanation.as_list()
    return jsonify({
        "features": [i[0] for i in items],
        "weights": [round(i[1], 4) for i in items],
        "horizon": horizon,
        "model_type": model_type,
    })
