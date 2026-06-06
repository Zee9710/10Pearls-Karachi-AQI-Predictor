from datetime import date, timedelta
from src.feature_pipeline import run_pipeline, write_to_feature_store


def backfill(months: int = 12) -> None:
    end = date.today() - timedelta(days=1)
    start = end - timedelta(days=30 * months)

    current = start
    chunk_days = 30

    while current < end:
        chunk_end = min(current + timedelta(days=chunk_days), end)
        print(f"Backfilling {current} → {chunk_end}")
        df = run_pipeline(current.isoformat(), chunk_end.isoformat())
        write_to_feature_store(df)
        current = chunk_end + timedelta(days=1)

    print("Backfill complete.")


if __name__ == "__main__":
    backfill(months=12)
