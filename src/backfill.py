"""Backfill historical (features, targets) into the feature store.

Runs the feature pipeline over a range of past dates and upserts the rows
into Supabase, generating the training data for the ML models. Run once to
seed the store, then the hourly pipeline keeps it current.
"""
from __future__ import annotations

from datetime import date, timedelta

from src.feature_pipeline import run_pipeline
from src.feature_store import upsert_features


def run_backfill(days: int = 365) -> int:
    # End 8 days back so the weather archive endpoint (vs forecast) is used and
    # the air-quality history is fully materialized.
    end = date.today() - timedelta(days=8)
    start = end - timedelta(days=days)
    df = run_pipeline(start.isoformat(), end.isoformat())
    n = upsert_features(df)
    print(f"Backfilled {n} feature rows ({start} -> {end})")
    return n


if __name__ == "__main__":
    run_backfill()
