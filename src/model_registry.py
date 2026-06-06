import os
import json
import joblib
import numpy as np
import pandas as pd
import hopsworks

from src.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
    HORIZONS, TABULAR_FEATURES, LSTM_RAW_FEATURES, LSTM_SEQ_LEN,
)
from src.aqi import aqi_category


def _get_project():
    return hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )


def load_best_model(project, horizon: int):
    mr = project.get_model_registry()
    hw_model = mr.get_best_model(f"aqi_best_{horizon}h", metric="rmse", direction="min")
    model_dir = hw_model.download()

    info_path = os.path.join(model_dir, "model_info.json")
    with open(info_path) as f:
        info = json.load(f)

    model_type = info["model_type"]

    if model_type == "lstm":
        import tensorflow as tf
        model = tf.keras.models.load_model(os.path.join(model_dir, "lstm_model"))
        scaler = joblib.load(os.path.join(model_dir, "lstm_scaler.pkl"))
    else:
        model = joblib.load(os.path.join(model_dir, "model.pkl"))
        scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))

    return model, scaler, model_type, info.get("all_metrics", {})


def predict_single(model, scaler, model_type: str, feature_row: pd.Series, sequence_df: pd.DataFrame) -> float:
    if model_type == "lstm":
        seq = sequence_df[LSTM_RAW_FEATURES].iloc[-LSTM_SEQ_LEN:].values.astype(np.float32)
        n_features = seq.shape[1]
        seq_scaled = scaler.transform(seq.reshape(-1, n_features)).reshape(1, LSTM_SEQ_LEN, n_features)
        return float(model.predict(seq_scaled, verbose=0).flatten()[0])
    else:
        X = feature_row[TABULAR_FEATURES].values.reshape(1, -1)
        X_scaled = scaler.transform(X)
        return float(model.predict(X_scaled)[0])


def predict_multi_horizon(current_df: pd.DataFrame) -> dict:
    """
    current_df: recent feature rows (at least LSTM_SEQ_LEN rows), sorted by datetime.
    Returns: {24: {...}, 48: {...}, 72: {...}} with predicted_aqi, category, datetime, all_metrics.
    """
    project = _get_project()
    latest_row = current_df.iloc[-1]
    latest_time = latest_row["datetime"]

    results = {}
    for h in HORIZONS:
        model, scaler, model_type, all_metrics = load_best_model(project, h)
        predicted_aqi = predict_single(model, scaler, model_type, latest_row, current_df)
        predicted_aqi = max(0.0, predicted_aqi)
        results[h] = {
            "predicted_aqi": round(predicted_aqi, 1),
            "category": aqi_category(int(predicted_aqi)),
            "datetime": latest_time + pd.Timedelta(hours=h),
            "all_metrics": all_metrics,
        }

    return results


def load_historical_features(days: int = 30) -> pd.DataFrame:
    project = _get_project()
    fs = project.get_feature_store()
    from src.config import FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION
    fg = fs.get_feature_group(FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    fv = fs.get_or_create_feature_view(
        name="aqi_dashboard_view",
        version=1,
        query=fg.select_all(),
    )
    df = fv.get_batch_data()
    df = df.sort_values("datetime").reset_index(drop=True)
    # Handle both tz-aware and tz-naive datetime columns
    if df["datetime"].dt.tz is not None:
        cutoff = pd.Timestamp.now(tz=df["datetime"].dt.tz) - pd.Timedelta(days=days)
    else:
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    return df[df["datetime"] >= cutoff].reset_index(drop=True)
