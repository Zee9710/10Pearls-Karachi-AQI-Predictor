import os
import json
import tempfile
import joblib
import numpy as np
import pandas as pd
import hopsworks
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import (
    HOPSWORKS_API_KEY, HOPSWORKS_PROJECT,
    FEATURE_GROUP_NAME, FEATURE_GROUP_VERSION,
    FEATURE_VIEW_NAME, FEATURE_VIEW_VERSION,
    HORIZONS, TABULAR_FEATURES, LSTM_RAW_FEATURES,
)


def _get_project():
    return hopsworks.login(
        api_key_value=HOPSWORKS_API_KEY,
        project=HOPSWORKS_PROJECT,
    )


def load_training_data() -> pd.DataFrame:
    # Direct fetch from Open-Meteo instead of the Hopsworks offline store, which
    # depends on a free-tier Spark materialization job that can stall for a long
    # time. end_date is offset 8 days so the weather archive endpoint (vs the
    # forecast endpoint) is used and the data is fully available.
    from datetime import date, timedelta
    from src.feature_pipeline import run_pipeline

    end = date.today() - timedelta(days=8)
    start = end - timedelta(days=365)
    df = run_pipeline(start.isoformat(), end.isoformat())
    return df.sort_values("datetime").reset_index(drop=True)


def _evaluate(model, X_test, y_test, scaler):
    preds = model.predict(scaler.transform(X_test))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    return {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }


def _register_sklearn_model(project, name, model, scaler, metrics, model_type_str, all_metrics=None):
    mr = project.get_model_registry()
    with tempfile.TemporaryDirectory() as tmpdir:
        joblib.dump(model, os.path.join(tmpdir, "model.pkl"))
        joblib.dump(scaler, os.path.join(tmpdir, "scaler.pkl"))
        info = {"model_type": model_type_str, "features": TABULAR_FEATURES}
        if all_metrics:
            info["all_metrics"] = all_metrics
        with open(os.path.join(tmpdir, "model_info.json"), "w") as f:
            json.dump(info, f)
        hw_model = mr.python.create_model(name=name, metrics=metrics)
        hw_model.save(tmpdir)


def _register_lstm_model(project, name, keras_model, scaler, metrics, all_metrics=None):
    mr = project.get_model_registry()
    with tempfile.TemporaryDirectory() as tmpdir:
        keras_model.save(os.path.join(tmpdir, "lstm_model"))
        joblib.dump(scaler, os.path.join(tmpdir, "lstm_scaler.pkl"))
        info = {"model_type": "lstm", "features": LSTM_RAW_FEATURES}
        if all_metrics:
            info["all_metrics"] = all_metrics
        with open(os.path.join(tmpdir, "model_info.json"), "w") as f:
            json.dump(info, f)
        hw_model = mr.tensorflow.create_model(name=name, metrics=metrics)
        hw_model.save(tmpdir)


def _build_lstm_sequences(df, horizon):
    from src.config import LSTM_SEQ_LEN, LSTM_RAW_FEATURES
    data = df[LSTM_RAW_FEATURES].values.astype(np.float32)
    X, y = [], []
    for i in range(LSTM_SEQ_LEN, len(data) - horizon):
        X.append(data[i - LSTM_SEQ_LEN:i])
        y.append(data[i + horizon - 1, -1])  # last feature is "aqi"
    return np.array(X), np.array(y)


def _train_lstm(df, horizon, split_idx):
    from src.config import LSTM_SEQ_LEN
    from sklearn.preprocessing import MinMaxScaler
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping

    X, y = _build_lstm_sequences(df, horizon)
    lstm_split = split_idx - LSTM_SEQ_LEN
    X_train, X_test = X[:lstm_split], X[lstm_split:]
    y_train, y_test = y[:lstm_split], y[lstm_split:]

    scaler = MinMaxScaler()
    n_features = X_train.shape[2]
    X_train_s = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(LSTM_SEQ_LEN, n_features)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    model.fit(
        X_train_s, y_train,
        epochs=50, batch_size=64,
        validation_split=0.1,
        callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0,
    )

    preds = model.predict(X_test_s, verbose=0).flatten()
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    metrics = {
        "rmse": rmse,
        "mae": float(mean_absolute_error(y_test, preds)),
        "r2": float(r2_score(y_test, preds)),
    }
    return model, scaler, metrics


def train_for_horizon(df: pd.DataFrame, horizon: int, project) -> dict:
    from xgboost import XGBRegressor
    from lightgbm import LGBMRegressor

    df = df.copy()
    df["target"] = df["aqi"].shift(-horizon)
    df = df.dropna(subset=["target"]).reset_index(drop=True)

    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

    X_train = train_df[TABULAR_FEATURES].values
    y_train = train_df["target"].values
    X_test = test_df[TABULAR_FEATURES].values
    y_test = test_df["target"].values

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)

    all_metrics = {}
    best_rmse = float("inf")
    best_name = None
    best_model = None
    best_scaler = None

    # Ridge
    ridge = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
    ridge.fit(X_train_s, y_train)
    m = _evaluate(ridge, X_test, y_test, scaler)
    all_metrics["ridge"] = m
    if m["rmse"] < best_rmse:
        best_rmse, best_name, best_model, best_scaler = m["rmse"], "ridge", ridge, scaler

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_s, y_train)
    m = _evaluate(rf, X_test, y_test, scaler)
    all_metrics["random_forest"] = m
    if m["rmse"] < best_rmse:
        best_rmse, best_name, best_model, best_scaler = m["rmse"], "random_forest", rf, scaler

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train_s, y_train)
    m = _evaluate(xgb, X_test, y_test, scaler)
    all_metrics["xgboost"] = m
    if m["rmse"] < best_rmse:
        best_rmse, best_name, best_model, best_scaler = m["rmse"], "xgboost", xgb, scaler

    # LightGBM
    lgbm = LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        subsample=0.8, colsample_bytree=0.8, random_state=42,
        verbose=-1,
    )
    lgbm.fit(X_train_s, y_train)
    m = _evaluate(lgbm, X_test, y_test, scaler)
    all_metrics["lightgbm"] = m
    if m["rmse"] < best_rmse:
        best_rmse, best_name, best_model, best_scaler = m["rmse"], "lightgbm", lgbm, scaler

    # LSTM
    lstm_model, lstm_scaler, m = _train_lstm(df, horizon, split_idx)
    all_metrics["lstm"] = m
    lstm_is_best = m["rmse"] < best_rmse
    if lstm_is_best:
        best_rmse, best_name = m["rmse"], "lstm"

    model_name = f"aqi_best_{horizon}h"
    if best_name == "lstm":
        _register_lstm_model(
            project, model_name, lstm_model, lstm_scaler,
            all_metrics["lstm"], all_metrics=all_metrics,
        )
    else:
        _register_sklearn_model(
            project, model_name, best_model, best_scaler,
            all_metrics[best_name], best_name, all_metrics=all_metrics,
        )

    print(f"  h={horizon}h  best={best_name}  RMSE={best_rmse:.2f}")
    return all_metrics


def run_training():
    project = _get_project()
    df = load_training_data()
    for h in HORIZONS:
        print(f"Training horizon +{h}h ...")
        train_for_horizon(df, h, project)


if __name__ == "__main__":
    run_training()
