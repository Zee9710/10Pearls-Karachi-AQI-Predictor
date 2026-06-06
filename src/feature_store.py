"""Supabase-backed feature store (via PostgREST).

Stores the computed (features, target) rows in a Postgres table through
Supabase's REST API. The feature pipeline upserts rows here, and the
training/serving code reads them back — replacing live API fetches with a
real feature store. Uses plain HTTP (requests) so it works with Supabase's
new `sb_secret_` API keys.
"""
from __future__ import annotations

import os
import requests
import pandas as pd

from src.config import TIMEZONE

TABLE = "aqi_features"
_PAGE = 1000


def _base() -> tuple[str, dict]:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key:
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY must be set to use the feature store."
        )
    endpoint = f"{url.rstrip('/')}/rest/v1/{TABLE}"
    headers = {"apikey": key, "Authorization": f"Bearer {key}"}
    return endpoint, headers


def _to_records(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    out["datetime"] = out["datetime"].apply(lambda t: pd.Timestamp(t).isoformat())
    records = out.to_dict(orient="records")
    for r in records:
        for k, v in r.items():
            if k != "datetime":
                r[k] = None if pd.isna(v) else float(v)
    return records


def _parse(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(TIMEZONE)
    return df.sort_values("datetime").reset_index(drop=True)


def upsert_features(df: pd.DataFrame, chunk: int = 500) -> int:
    """Insert/update feature rows keyed by datetime. Returns row count written."""
    if df.empty:
        return 0
    endpoint, headers = _base()
    headers = {
        **headers,
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates,return=minimal",
    }
    records = _to_records(df)
    for i in range(0, len(records), chunk):
        resp = requests.post(
            endpoint, headers=headers, json=records[i : i + chunk], timeout=60
        )
        resp.raise_for_status()
    return len(records)


def read_features() -> pd.DataFrame:
    """Read the full feature table (paginated), sorted by datetime ascending."""
    endpoint, headers = _base()
    rows: list[dict] = []
    offset = 0
    while True:
        resp = requests.get(
            endpoint,
            headers=headers,
            params={
                "select": "*",
                "order": "datetime.asc",
                "limit": _PAGE,
                "offset": offset,
            },
            timeout=60,
        )
        resp.raise_for_status()
        batch = resp.json()
        rows.extend(batch)
        if len(batch) < _PAGE:
            break
        offset += _PAGE
    return _parse(rows)


def read_recent(n: int = 200) -> pd.DataFrame:
    """Read the newest n rows, returned sorted by datetime ascending."""
    endpoint, headers = _base()
    resp = requests.get(
        endpoint,
        headers=headers,
        params={"select": "*", "order": "datetime.desc", "limit": n},
        timeout=60,
    )
    resp.raise_for_status()
    return _parse(resp.json())
