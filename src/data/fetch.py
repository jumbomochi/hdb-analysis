"""Fetch HDB resale flat prices from data.gov.sg API."""

import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://data.gov.sg/api/action/datastore_search"

DATASETS = {
    "2017_onwards": {
        "id": "d_8b84c4ee58e3cfc0ece0d773c8ca6abc",
        "description": "Jan 2017 onwards (registration date)",
    },
    "2015_2016": {
        "id": "d_ea9ed51da2787afaf8e51f827c304208",
        "description": "Jan 2015 – Dec 2016 (registration date)",
    },
    "2012_2014": {
        "id": "d_2d5ff9ea31397b66239f245f57751537",
        "description": "Mar 2012 – Dec 2014 (registration date)",
    },
}


def fetch_dataset(dataset_id: str, limit: int = 5000, max_retries: int = 5) -> list[dict]:
    """Fetch all records from a single dataset via pagination."""
    records = []
    offset = 0

    while True:
        for attempt in range(max_retries):
            resp = requests.get(
                BASE_URL,
                params={"resource_id": dataset_id, "limit": limit, "offset": offset},
                timeout=60,
            )
            if resp.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"  Rate limited, waiting {wait}s...")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            break
        else:
            resp.raise_for_status()

        result = resp.json()["result"]
        batch = result["records"]
        records.extend(batch)
        print(f"  Fetched {len(records)}/{result['total']} records...")

        if offset + limit >= result["total"]:
            break
        offset += limit
        time.sleep(1)

    return records


def fetch_all_resale_data(output_dir: str = "data/raw") -> Path:
    """Fetch all datasets and save as a single combined CSV."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_frames = []
    for name, info in DATASETS.items():
        print(f"Fetching {name}: {info['description']}...")
        records = fetch_dataset(info["id"])
        df = pd.DataFrame(records)
        df["dataset"] = name
        all_frames.append(df)
        print(f"  → {len(df)} records")

    combined = pd.concat(all_frames, ignore_index=True)
    csv_path = output_path / "resale_prices.csv"
    combined.to_csv(csv_path, index=False)
    print(f"Saved {len(combined)} total records to {csv_path}")
    return csv_path


if __name__ == "__main__":
    fetch_all_resale_data()
