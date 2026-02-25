# HDB Resale Flat Valuation Dashboard — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an interactive Streamlit dashboard that helps a prospective 5-room HDB buyer evaluate resale flat valuations — comparing towns, understanding price drivers, estimating fair prices, and evaluating specific listings.

**Architecture:** Python Streamlit app with 4 pages. Data fetched from data.gov.sg API, enriched with MRT proximity via OneMap geocoding. XGBoost model trained on recent transactions for price estimation with quantile-based confidence intervals.

**Tech Stack:** Python 3.11+, Streamlit, pandas, plotly, XGBoost, scikit-learn, requests

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/model/__init__.py`
- Create: `src/utils/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/data/__init__.py`
- Create: `tests/model/__init__.py`
- Create: `tests/utils/__init__.py`
- Create: `.gitignore`

**Step 1: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
data/raw/
data/processed/
models/*.joblib
.streamlit/
*.egg-info/
```

**Step 2: Create requirements.txt**

```
streamlit>=1.30.0
pandas>=2.1.0
plotly>=5.18.0
xgboost>=2.0.0
scikit-learn>=1.3.0
requests>=2.31.0
joblib>=1.3.0
pytest>=7.4.0
```

**Step 3: Create directory structure**

```bash
mkdir -p data/raw data/processed data/reference models
mkdir -p src/data src/model src/utils tests/data tests/model tests/utils
mkdir -p app/pages
touch src/__init__.py src/data/__init__.py src/model/__init__.py src/utils/__init__.py
touch tests/__init__.py tests/data/__init__.py tests/model/__init__.py tests/utils/__init__.py
```

**Step 4: Create virtual environment and install dependencies**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Step 5: Commit**

```bash
git add .gitignore requirements.txt src/ tests/
git commit -m "chore: project scaffold with dependencies and directory structure"
```

---

### Task 2: Data Fetch Script

**Files:**
- Create: `src/data/fetch.py`
- Create: `tests/data/test_fetch.py`

**Step 1: Write tests for fetch utilities**

```python
# tests/data/test_fetch.py
import pytest
from unittest.mock import patch, MagicMock
from src.data.fetch import fetch_dataset, fetch_all_resale_data, DATASETS


def test_datasets_constant_has_expected_keys():
    """All dataset entries should have id and description."""
    for name, info in DATASETS.items():
        assert "id" in info
        assert "description" in info


def test_fetch_dataset_paginates(tmp_path):
    """fetch_dataset should paginate through all records."""
    mock_response_page1 = MagicMock()
    mock_response_page1.json.return_value = {
        "result": {
            "total": 3,
            "records": [
                {"month": "2024-01", "town": "TAMPINES", "resale_price": "500000"},
                {"month": "2024-01", "town": "BEDOK", "resale_price": "450000"},
            ],
        }
    }
    mock_response_page1.raise_for_status = MagicMock()

    mock_response_page2 = MagicMock()
    mock_response_page2.json.return_value = {
        "result": {
            "total": 3,
            "records": [
                {"month": "2024-02", "town": "WOODLANDS", "resale_price": "400000"},
            ],
        }
    }
    mock_response_page2.raise_for_status = MagicMock()

    with patch("src.data.fetch.requests.get", side_effect=[mock_response_page1, mock_response_page2]):
        records = fetch_dataset("d_test_id", limit=2)
        assert len(records) == 3
        assert records[0]["town"] == "TAMPINES"
        assert records[2]["town"] == "WOODLANDS"


def test_fetch_all_resale_data_saves_csv(tmp_path):
    """fetch_all_resale_data should save combined CSV to output_dir."""
    sample_records = [
        {"month": "2024-01", "town": "TAMPINES", "flat_type": "5 ROOM",
         "block": "123", "street_name": "TAMPINES ST 21", "storey_range": "07 TO 09",
         "floor_area_sqm": "110", "flat_model": "Improved", "lease_commence_date": "1990",
         "remaining_lease": "63 years 05 months", "resale_price": "500000"},
    ]

    with patch("src.data.fetch.fetch_dataset", return_value=sample_records):
        output_path = fetch_all_resale_data(output_dir=str(tmp_path))
        assert output_path.exists()
        import pandas as pd
        df = pd.read_csv(output_path)
        assert len(df) == len(DATASETS)  # One record per dataset call
        assert "town" in df.columns
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/data/test_fetch.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.data.fetch'`

**Step 3: Implement fetch.py**

```python
# src/data/fetch.py
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


def fetch_dataset(dataset_id: str, limit: int = 1000) -> list[dict]:
    """Fetch all records from a single dataset via pagination."""
    records = []
    offset = 0

    while True:
        resp = requests.get(
            BASE_URL,
            params={"resource_id": dataset_id, "limit": limit, "offset": offset},
            timeout=30,
        )
        resp.raise_for_status()
        result = resp.json()["result"]
        batch = result["records"]
        records.extend(batch)

        if offset + limit >= result["total"]:
            break
        offset += limit
        time.sleep(0.3)

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
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/data/test_fetch.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/fetch.py tests/data/test_fetch.py
git commit -m "feat: add data fetch script for HDB resale prices from data.gov.sg"
```

---

### Task 3: MRT Station Reference Data

**Files:**
- Create: `src/utils/mrt.py`
- Create: `tests/utils/test_mrt.py`

**Step 1: Write tests for MRT utilities**

```python
# tests/utils/test_mrt.py
import pytest
from unittest.mock import patch, MagicMock
from src.utils.mrt import (
    fetch_mrt_stations,
    haversine_distance,
    find_nearest_mrt,
)


def test_haversine_distance_known_values():
    """Test haversine with known Singapore distances."""
    # Tampines MRT to Pasir Ris MRT ≈ 2.5 km
    dist = haversine_distance(1.3529, 103.9452, 1.3731, 103.9493)
    assert 2.0 < dist < 3.0


def test_haversine_distance_same_point():
    dist = haversine_distance(1.35, 103.94, 1.35, 103.94)
    assert dist == 0.0


def test_find_nearest_mrt():
    """Given a point and a list of stations, return closest station and distance."""
    stations = [
        {"name": "Tampines", "lat": 1.3529, "lng": 103.9452},
        {"name": "Pasir Ris", "lat": 1.3731, "lng": 103.9493},
    ]
    name, dist = find_nearest_mrt(1.3540, 103.9460, stations)
    assert name == "Tampines"
    assert dist < 0.5  # very close to Tampines


def test_fetch_mrt_stations_parses_geojson(tmp_path):
    """fetch_mrt_stations should parse GeoJSON exits and aggregate to station centroids."""
    mock_geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [103.85, 1.35]},
                "properties": {"STATION_NA": "Toa Payoh", "EXIT_CODE": "Exit A"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [103.851, 1.351]},
                "properties": {"STATION_NA": "Toa Payoh", "EXIT_CODE": "Exit B"},
            },
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [103.86, 1.36]},
                "properties": {"STATION_NA": "Braddell", "EXIT_CODE": "Exit A"},
            },
        ],
    }

    mock_poll = MagicMock()
    mock_poll.json.return_value = {"data": {"url": "https://example.com/data.geojson"}}
    mock_poll.raise_for_status = MagicMock()

    mock_download = MagicMock()
    mock_download.json.return_value = mock_geojson
    mock_download.raise_for_status = MagicMock()

    with patch("src.utils.mrt.requests.get", side_effect=[mock_poll, mock_download]):
        stations = fetch_mrt_stations(cache_path=str(tmp_path / "mrt.json"))

    assert len(stations) == 2
    names = {s["name"] for s in stations}
    assert names == {"Toa Payoh", "Braddell"}

    # Toa Payoh should be the average of its two exits
    tp = next(s for s in stations if s["name"] == "Toa Payoh")
    assert abs(tp["lat"] - 1.3505) < 0.001
    assert abs(tp["lng"] - 103.8505) < 0.001
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/utils/test_mrt.py -v
```
Expected: FAIL

**Step 3: Implement mrt.py**

```python
# src/utils/mrt.py
"""MRT station data and distance calculations."""

import json
import math
from pathlib import Path

import requests

MRT_EXITS_DATASET_ID = "d_b39d3a0871985372d7e1637193335da5"
POLL_URL = f"https://api-open.data.gov.sg/v1/public/api/datasets/{MRT_EXITS_DATASET_ID}/poll-download"


def haversine_distance(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Calculate distance in km between two lat/lng points."""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlng / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def find_nearest_mrt(
    lat: float, lng: float, stations: list[dict]
) -> tuple[str, float]:
    """Return (station_name, distance_km) of nearest MRT station."""
    best_name = ""
    best_dist = float("inf")
    for s in stations:
        d = haversine_distance(lat, lng, s["lat"], s["lng"])
        if d < best_dist:
            best_dist = d
            best_name = s["name"]
    return best_name, best_dist


def fetch_mrt_stations(cache_path: str = "data/reference/mrt_stations.json") -> list[dict]:
    """Fetch MRT station exit GeoJSON from data.gov.sg and aggregate to station centroids."""
    cache = Path(cache_path)
    if cache.exists():
        with open(cache) as f:
            return json.load(f)

    # Poll for download URL
    resp = requests.get(POLL_URL, timeout=30)
    resp.raise_for_status()
    download_url = resp.json()["data"]["url"]

    # Download GeoJSON
    resp = requests.get(download_url, timeout=60)
    resp.raise_for_status()
    geojson = resp.json()

    # Aggregate exits to station centroids
    station_coords: dict[str, list[tuple[float, float]]] = {}
    for feature in geojson["features"]:
        name = feature["properties"]["STATION_NA"]
        lng, lat = feature["geometry"]["coordinates"]
        station_coords.setdefault(name, []).append((lat, lng))

    stations = []
    for name, coords in sorted(station_coords.items()):
        avg_lat = sum(c[0] for c in coords) / len(coords)
        avg_lng = sum(c[1] for c in coords) / len(coords)
        stations.append({"name": name, "lat": avg_lat, "lng": avg_lng})

    # Cache locally
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(stations, f, indent=2)

    return stations
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/utils/test_mrt.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/utils/mrt.py tests/utils/test_mrt.py
git commit -m "feat: add MRT station fetcher and distance utilities"
```

---

### Task 4: OneMap Geocoding

**Files:**
- Create: `src/data/geocode.py`
- Create: `tests/data/test_geocode.py`

**Step 1: Write tests for geocoding**

```python
# tests/data/test_geocode.py
import pytest
from unittest.mock import patch, MagicMock
from src.data.geocode import geocode_address, batch_geocode_blocks


def test_geocode_address_returns_coords():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "found": 1,
        "results": [
            {
                "LATITUDE": "1.3694",
                "LONGITUDE": "103.8490",
                "ADDRESS": "123 ANG MO KIO AVENUE 3 SINGAPORE 560123",
            }
        ],
    }
    mock_resp.raise_for_status = MagicMock()

    with patch("src.data.geocode.requests.get", return_value=mock_resp):
        result = geocode_address("123 ANG MO KIO AVE 3")

    assert result is not None
    assert abs(result["lat"] - 1.3694) < 0.001
    assert abs(result["lng"] - 103.849) < 0.001


def test_geocode_address_not_found():
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"found": 0, "results": []}
    mock_resp.raise_for_status = MagicMock()

    with patch("src.data.geocode.requests.get", return_value=mock_resp):
        result = geocode_address("NONEXISTENT ADDRESS")

    assert result is None


def test_batch_geocode_deduplicates():
    """Should only call geocode once per unique block+street."""
    blocks = [
        {"block": "123", "street_name": "TAMPINES ST 21"},
        {"block": "123", "street_name": "TAMPINES ST 21"},
        {"block": "456", "street_name": "BEDOK NORTH AVE 1"},
    ]

    call_count = 0

    def mock_geocode(addr):
        nonlocal call_count
        call_count += 1
        return {"lat": 1.35, "lng": 103.94}

    with patch("src.data.geocode.geocode_address", side_effect=mock_geocode):
        results = batch_geocode_blocks(blocks, delay=0)

    assert call_count == 2  # Only 2 unique addresses
    assert len(results) == 2
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/data/test_geocode.py -v
```
Expected: FAIL

**Step 3: Implement geocode.py**

```python
# src/data/geocode.py
"""Geocode HDB addresses using Singapore OneMap API."""

import json
import time
from pathlib import Path

import requests

ONEMAP_URL = "https://www.onemap.gov.sg/api/common/elastic/search"


def geocode_address(address: str) -> dict | None:
    """Geocode a single address. Returns {lat, lng} or None if not found."""
    resp = requests.get(
        ONEMAP_URL,
        params={"searchVal": address, "returnGeom": "Y", "getAddrDetails": "Y", "pageNum": 1},
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()

    if data.get("found", 0) == 0:
        return None

    result = data["results"][0]
    return {
        "lat": float(result["LATITUDE"]),
        "lng": float(result["LONGITUDE"]),
    }


def batch_geocode_blocks(
    blocks: list[dict],
    cache_path: str = "data/reference/block_coords.json",
    delay: float = 0.3,
) -> dict[str, dict]:
    """Geocode unique block+street combos. Returns {address_key: {lat, lng}}."""
    cache = Path(cache_path)
    cached = {}
    if cache.exists():
        with open(cache) as f:
            cached = json.load(f)

    # Deduplicate
    unique_addresses = {}
    for b in blocks:
        key = f"{b['block']} {b['street_name']}"
        unique_addresses[key] = key

    results = dict(cached)
    new_count = 0
    for key in unique_addresses:
        if key in results:
            continue
        coords = geocode_address(key)
        if coords:
            results[key] = coords
            new_count += 1
        if delay > 0:
            time.sleep(delay)

        # Save periodically
        if new_count % 100 == 0 and new_count > 0:
            cache.parent.mkdir(parents=True, exist_ok=True)
            with open(cache, "w") as f:
                json.dump(results, f)
            print(f"  Geocoded {new_count} new addresses...")

    # Final save
    if new_count > 0:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "w") as f:
            json.dump(results, f)
        print(f"  Geocoded {new_count} new addresses total")

    return results
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/data/test_geocode.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/geocode.py tests/data/test_geocode.py
git commit -m "feat: add OneMap geocoding for HDB blocks"
```

---

### Task 5: Data Processing & Feature Engineering

**Files:**
- Create: `src/data/process.py`
- Create: `tests/data/test_process.py`

**Step 1: Write tests for processing functions**

```python
# tests/data/test_process.py
import pytest
import pandas as pd
from src.data.process import (
    parse_storey_range,
    calculate_remaining_lease,
    add_price_per_sqm,
    process_raw_data,
)


def test_parse_storey_range():
    assert parse_storey_range("07 TO 09") == 8
    assert parse_storey_range("01 TO 03") == 2
    assert parse_storey_range("10 TO 12") == 11
    assert parse_storey_range("40 TO 42") == 41


def test_calculate_remaining_lease_from_commence_date():
    """99-year lease from 1990, checked in 2024 → 65 years remaining."""
    remaining = calculate_remaining_lease(1990, 2024)
    assert remaining == 65


def test_calculate_remaining_lease_from_string():
    """Parse '61 years 04 months' format."""
    remaining = calculate_remaining_lease(None, None, remaining_str="61 years 04 months")
    assert abs(remaining - 61.33) < 0.1


def test_add_price_per_sqm():
    df = pd.DataFrame({"resale_price": [500000, 600000], "floor_area_sqm": [100, 120]})
    result = add_price_per_sqm(df)
    assert list(result["price_per_sqm"]) == [5000.0, 5000.0]


def test_process_raw_data_produces_expected_columns(tmp_path):
    """Full processing pipeline should produce all expected columns."""
    raw = pd.DataFrame(
        {
            "month": ["2024-01"],
            "town": ["TAMPINES"],
            "flat_type": ["5 ROOM"],
            "block": ["123"],
            "street_name": ["TAMPINES ST 21"],
            "storey_range": ["07 TO 09"],
            "floor_area_sqm": [110],
            "flat_model": ["Improved"],
            "lease_commence_date": [1990],
            "remaining_lease": ["65 years 00 months"],
            "resale_price": [500000],
        }
    )
    raw_path = tmp_path / "raw.csv"
    raw.to_csv(raw_path, index=False)

    result = process_raw_data(
        raw_path,
        block_coords={"123 TAMPINES ST 21": {"lat": 1.35, "lng": 103.94}},
        mrt_stations=[{"name": "Tampines", "lat": 1.3529, "lng": 103.9452}],
    )

    expected_cols = [
        "month", "town", "flat_type", "block", "street_name", "storey_range",
        "floor_area_sqm", "flat_model", "lease_commence_date", "resale_price",
        "year", "quarter", "storey_mid", "remaining_lease_years",
        "price_per_sqm", "lat", "lng", "nearest_mrt", "nearest_mrt_dist_km",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/data/test_process.py -v
```
Expected: FAIL

**Step 3: Implement process.py**

```python
# src/data/process.py
"""Clean and feature-engineer HDB resale data."""

import re
from pathlib import Path

import pandas as pd

from src.utils.mrt import find_nearest_mrt


def parse_storey_range(storey_str: str) -> int:
    """Parse '07 TO 09' into midpoint 8."""
    parts = storey_str.split(" TO ")
    low, high = int(parts[0]), int(parts[1])
    return (low + high) // 2


def calculate_remaining_lease(
    commence_year: int | None = None,
    reference_year: int | None = None,
    remaining_str: str | None = None,
    lease_duration: int = 99,
) -> float:
    """Calculate remaining lease in years.

    Either from commence_year + reference_year, or by parsing a string like '61 years 04 months'.
    """
    if remaining_str:
        match = re.match(r"(\d+)\s*years?\s*(?:(\d+)\s*months?)?", remaining_str)
        if match:
            years = int(match.group(1))
            months = int(match.group(2)) if match.group(2) else 0
            return years + months / 12
    if commence_year and reference_year:
        return lease_duration - (reference_year - commence_year)
    return 0.0


def add_price_per_sqm(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_per_sqm column."""
    df = df.copy()
    df["price_per_sqm"] = df["resale_price"] / df["floor_area_sqm"]
    return df


def process_raw_data(
    raw_csv_path: str | Path,
    block_coords: dict[str, dict] | None = None,
    mrt_stations: list[dict] | None = None,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Full processing pipeline: clean, feature-engineer, enrich with geo data."""
    df = pd.read_csv(raw_csv_path)

    # Type conversions
    df["resale_price"] = pd.to_numeric(df["resale_price"], errors="coerce")
    df["floor_area_sqm"] = pd.to_numeric(df["floor_area_sqm"], errors="coerce")
    df["lease_commence_date"] = pd.to_numeric(df["lease_commence_date"], errors="coerce")

    # Date features
    df["year"] = df["month"].str[:4].astype(int)
    df["quarter"] = pd.to_datetime(df["month"]).dt.quarter

    # Storey midpoint
    df["storey_mid"] = df["storey_range"].apply(parse_storey_range)

    # Remaining lease
    if "remaining_lease" in df.columns:
        df["remaining_lease_years"] = df.apply(
            lambda row: calculate_remaining_lease(
                commence_year=row["lease_commence_date"],
                reference_year=row["year"],
                remaining_str=row.get("remaining_lease"),
            ),
            axis=1,
        )
    else:
        df["remaining_lease_years"] = df.apply(
            lambda row: calculate_remaining_lease(
                commence_year=row["lease_commence_date"],
                reference_year=row["year"],
            ),
            axis=1,
        )

    # Price per sqm
    df = add_price_per_sqm(df)

    # Geocoding enrichment
    if block_coords:
        df["address_key"] = df["block"] + " " + df["street_name"]
        df["lat"] = df["address_key"].map(lambda k: block_coords.get(k, {}).get("lat"))
        df["lng"] = df["address_key"].map(lambda k: block_coords.get(k, {}).get("lng"))
        df.drop(columns=["address_key"], inplace=True)
    else:
        df["lat"] = None
        df["lng"] = None

    # MRT distance
    if mrt_stations and block_coords:
        def _nearest(row):
            if pd.isna(row["lat"]) or pd.isna(row["lng"]):
                return pd.Series({"nearest_mrt": None, "nearest_mrt_dist_km": None})
            name, dist = find_nearest_mrt(row["lat"], row["lng"], mrt_stations)
            return pd.Series({"nearest_mrt": name, "nearest_mrt_dist_km": round(dist, 3)})

        mrt_info = df.apply(_nearest, axis=1)
        df = pd.concat([df, mrt_info], axis=1)
    else:
        df["nearest_mrt"] = None
        df["nearest_mrt_dist_km"] = None

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/data/test_process.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/data/process.py tests/data/test_process.py
git commit -m "feat: add data processing and feature engineering pipeline"
```

---

### Task 6: XGBoost Model Training

**Files:**
- Create: `src/model/train.py`
- Create: `tests/model/test_train.py`

**Step 1: Write tests for model training**

```python
# tests/model/test_train.py
import pytest
import numpy as np
import pandas as pd
from src.model.train import prepare_features, train_model


def _make_sample_data(n=200):
    rng = np.random.RandomState(42)
    towns = rng.choice(["TAMPINES", "BEDOK", "WOODLANDS"], n)
    floor_area = rng.uniform(90, 130, n)
    storey = rng.randint(1, 30, n)
    lease = rng.uniform(50, 95, n)
    mrt_dist = rng.uniform(0.1, 2.0, n)
    price = 3000 * floor_area + 5000 * storey + 2000 * lease - 50000 * mrt_dist + rng.normal(0, 10000, n)
    return pd.DataFrame({
        "town": towns,
        "floor_area_sqm": floor_area,
        "storey_mid": storey,
        "remaining_lease_years": lease,
        "nearest_mrt_dist_km": mrt_dist,
        "resale_price": price,
    })


def test_prepare_features_returns_expected_shape():
    df = _make_sample_data()
    X, y, feature_names = prepare_features(df)
    assert X.shape[0] == len(df)
    assert len(y) == len(df)
    assert "floor_area_sqm" in feature_names
    assert any("town_" in f for f in feature_names)  # one-hot encoded


def test_train_model_returns_fitted_model():
    df = _make_sample_data()
    X, y, feature_names = prepare_features(df)
    model = train_model(X, y)
    preds = model.predict(X[:5])
    assert len(preds) == 5
    assert all(p > 0 for p in preds)


def test_train_model_reasonable_accuracy():
    """Model should achieve R² > 0.8 on this synthetic data."""
    df = _make_sample_data(500)
    X, y, feature_names = prepare_features(df)
    model = train_model(X, y)
    from sklearn.metrics import r2_score
    preds = model.predict(X)
    r2 = r2_score(y, preds)
    assert r2 > 0.8
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_train.py -v
```
Expected: FAIL

**Step 3: Implement train.py**

```python
# src/model/train.py
"""Train XGBoost model for HDB resale price prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare feature matrix and target from processed dataframe.

    Returns (X, y, feature_names).
    """
    feature_cols = ["floor_area_sqm", "storey_mid", "remaining_lease_years", "nearest_mrt_dist_km"]
    df_clean = df.dropna(subset=feature_cols + ["resale_price"]).copy()

    # One-hot encode town
    town_dummies = pd.get_dummies(df_clean["town"], prefix="town", dtype=float)
    numeric = df_clean[feature_cols].astype(float)
    X_df = pd.concat([numeric, town_dummies], axis=1)

    feature_names = list(X_df.columns)
    X = X_df.values
    y = df_clean["resale_price"].values.astype(float)

    return X, y, feature_names


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> XGBRegressor:
    """Train an XGBoost regressor."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[float, XGBRegressor]:
    """Train separate models for each quantile (for confidence intervals)."""
    models = {}
    for q in quantiles:
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            objective="reg:quantileerror",
            quantile_alpha=q,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        models[q] = model
    return models


def save_model(model, feature_names: list[str], path: str = "models/xgb_model.joblib"):
    """Save model and feature names."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, path)


def train_and_save(df: pd.DataFrame, models_dir: str = "models"):
    """Full training pipeline: prepare, train median + quantile models, save."""
    X, y, feature_names = prepare_features(df)

    print(f"Training on {len(y)} samples with {len(feature_names)} features...")

    # Train median model
    median_model = train_model(X, y)
    save_model(median_model, feature_names, f"{models_dir}/xgb_median.joblib")

    # Train quantile models for confidence intervals
    quantile_models = train_quantile_models(X, y)
    for q, model in quantile_models.items():
        save_model(model, feature_names, f"{models_dir}/xgb_q{q}.joblib")

    # Evaluate
    from sklearn.metrics import r2_score, mean_absolute_error
    preds = median_model.predict(X)
    print(f"Training R²: {r2_score(y, preds):.4f}")
    print(f"Training MAE: ${mean_absolute_error(y, preds):,.0f}")

    return median_model, feature_names
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/model/test_train.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/train.py tests/model/test_train.py
git commit -m "feat: add XGBoost model training with quantile regression"
```

---

### Task 7: Prediction Module

**Files:**
- Create: `src/model/predict.py`
- Create: `tests/model/test_predict.py`

**Step 1: Write tests for prediction**

```python
# tests/model/test_predict.py
import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.model.predict import predict_price, find_comparable_transactions


def test_find_comparable_transactions():
    df = pd.DataFrame({
        "town": ["TAMPINES"] * 5 + ["BEDOK"] * 3,
        "flat_type": ["5 ROOM"] * 8,
        "floor_area_sqm": [110, 112, 108, 115, 105, 110, 112, 108],
        "storey_mid": [8, 10, 5, 15, 3, 8, 10, 12],
        "remaining_lease_years": [65, 63, 70, 55, 80, 65, 63, 70],
        "resale_price": [500000, 510000, 480000, 530000, 470000, 490000, 500000, 480000],
        "month": ["2024-01"] * 8,
    })

    comps = find_comparable_transactions(
        df, town="TAMPINES", flat_type="5 ROOM", floor_area=110, storey=8, n=3
    )

    assert len(comps) <= 3
    assert all(comps["town"] == "TAMPINES")
```

**Step 2: Run tests to verify they fail**

```bash
pytest tests/model/test_predict.py -v
```
Expected: FAIL

**Step 3: Implement predict.py**

```python
# src/model/predict.py
"""Price prediction and comparable transaction lookup."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_models(models_dir: str = "models") -> dict:
    """Load all trained models."""
    result = {}
    models_path = Path(models_dir)

    median_path = models_path / "xgb_median.joblib"
    if median_path.exists():
        data = joblib.load(median_path)
        result["median"] = data["model"]
        result["feature_names"] = data["feature_names"]

    for q in [0.1, 0.5, 0.9]:
        q_path = models_path / f"xgb_q{q}.joblib"
        if q_path.exists():
            data = joblib.load(q_path)
            result[f"q{q}"] = data["model"]

    return result


def predict_price(
    models: dict,
    town: str,
    floor_area: float,
    storey: int,
    remaining_lease: float,
    mrt_dist: float,
) -> dict:
    """Predict resale price with confidence interval.

    Returns {estimate, low, high} where low/high are 10th/90th percentiles.
    """
    feature_names = models["feature_names"]

    # Build feature vector
    features = {
        "floor_area_sqm": floor_area,
        "storey_mid": storey,
        "remaining_lease_years": remaining_lease,
        "nearest_mrt_dist_km": mrt_dist,
    }

    # One-hot encode town
    for fname in feature_names:
        if fname.startswith("town_"):
            features[fname] = 1.0 if fname == f"town_{town}" else 0.0

    X = np.array([[features.get(f, 0.0) for f in feature_names]])

    result = {"estimate": float(models["median"].predict(X)[0])}

    if "q0.1" in models:
        result["low"] = float(models["q0.1"].predict(X)[0])
    if "q0.9" in models:
        result["high"] = float(models["q0.9"].predict(X)[0])

    return result


def find_comparable_transactions(
    df: pd.DataFrame,
    town: str,
    flat_type: str,
    floor_area: float,
    storey: int,
    n: int = 10,
    months_back: int = 12,
) -> pd.DataFrame:
    """Find the most similar recent transactions."""
    filtered = df[
        (df["town"] == town)
        & (df["flat_type"] == flat_type)
    ].copy()

    if filtered.empty:
        return filtered

    # Sort by recent first
    filtered = filtered.sort_values("month", ascending=False)

    # Limit to recent months
    if months_back and "month" in filtered.columns:
        cutoff = pd.Timestamp.now() - pd.DateOffset(months=months_back)
        cutoff_str = cutoff.strftime("%Y-%m")
        filtered = filtered[filtered["month"] >= cutoff_str]

    # Score by similarity (lower = more similar)
    filtered["_similarity"] = (
        abs(filtered["floor_area_sqm"] - floor_area) / 10
        + abs(filtered["storey_mid"] - storey) / 5
    )
    filtered = filtered.sort_values("_similarity").head(n)
    return filtered.drop(columns=["_similarity"])
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/model/test_predict.py -v
```
Expected: PASS

**Step 5: Commit**

```bash
git add src/model/predict.py tests/model/test_predict.py
git commit -m "feat: add prediction module with confidence intervals and comparable lookup"
```

---

### Task 8: Streamlit Home Page

**Files:**
- Create: `app/Home.py`

**Step 1: Create the Streamlit entry point**

```python
# app/Home.py
"""HDB Resale Flat Valuation Dashboard — Home page."""

import streamlit as st

st.set_page_config(
    page_title="HDB Resale Valuation",
    page_icon="🏠",
    layout="wide",
)

st.title("HDB Resale Flat Valuation Dashboard")
st.markdown(
    """
    An interactive tool to help you evaluate HDB resale flat prices in Singapore.

    ### Pages

    - **Town Comparison** — Compare median prices, trends, and value across towns
    - **Price Drivers** — Understand which factors impact resale prices the most
    - **Fair Price Estimator** — Get an estimated fair price for a flat based on its attributes
    - **Listing Evaluator** — Evaluate a specific listing against the model and recent transactions

    ### Data

    Resale transaction data from [data.gov.sg](https://data.gov.sg/collections/189/view),
    enriched with MRT proximity from OneMap.

    Use the sidebar to navigate between pages.
    """
)
```

**Step 2: Test manually**

```bash
cd /Users/huiliang/GitHub/hdb-analysis && streamlit run app/Home.py
```
Expected: Dashboard loads in browser with the home page content.

**Step 3: Commit**

```bash
git add app/Home.py
git commit -m "feat: add Streamlit home page"
```

---

### Task 9: Shared Data Loading Utilities

**Files:**
- Create: `app/data_loader.py`

This shared module loads the processed data and models once, cached by Streamlit.

**Step 1: Create data_loader.py**

```python
# app/data_loader.py
"""Shared data loading for Streamlit pages."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.predict import load_models


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Load the processed resale data."""
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "resale_processed.csv"
    if not path.exists():
        st.error(f"Processed data not found at {path}. Run the data pipeline first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_prediction_models() -> dict:
    """Load trained models."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models = load_models(str(models_dir))
    if "median" not in models:
        st.error("Trained model not found. Run the training pipeline first.")
        st.stop()
    return models


def get_towns(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique towns."""
    return sorted(df["town"].unique().tolist())


def get_flat_types(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique flat types."""
    return sorted(df["flat_type"].unique().tolist())
```

**Step 2: Commit**

```bash
git add app/data_loader.py
git commit -m "feat: add shared Streamlit data loading with caching"
```

---

### Task 10: Page 1 — Town Comparison

**Files:**
- Create: `app/pages/1_Town_Comparison.py`

**Step 1: Implement the Town Comparison page**

```python
# app/pages/1_Town_Comparison.py
"""Town Comparison — compare resale prices across towns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data_loader import load_processed_data, get_towns, get_flat_types

st.set_page_config(page_title="Town Comparison", layout="wide")
st.title("Town Comparison")

df = load_processed_data()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    flat_type = st.selectbox("Flat Type", get_flat_types(df), index=get_flat_types(df).index("5 ROOM") if "5 ROOM" in get_flat_types(df) else 0)
    year_range = st.slider(
        "Year Range",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].max()) - 3, int(df["year"].max())),
    )
    all_towns = get_towns(df)
    selected_towns = st.multiselect("Towns (leave empty for all)", all_towns)

# Filter data
filtered = df[
    (df["flat_type"] == flat_type)
    & (df["year"] >= year_range[0])
    & (df["year"] <= year_range[1])
]
if selected_towns:
    filtered = filtered[filtered["town"].isin(selected_towns)]

if filtered.empty:
    st.warning("No data matching filters.")
    st.stop()

# Median price by town
col1, col2 = st.columns(2)

with col1:
    st.subheader("Median Resale Price by Town")
    town_median = filtered.groupby("town")["resale_price"].median().sort_values(ascending=False).reset_index()
    fig = px.bar(town_median, x="town", y="resale_price", labels={"resale_price": "Median Price (SGD)", "town": "Town"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Price per sqm Distribution by Town")
    fig = px.box(filtered, x="town", y="price_per_sqm", labels={"price_per_sqm": "Price/sqm (SGD)", "town": "Town"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Price trend
st.subheader("Price Trend Over Time")
monthly = filtered.groupby(["month", "town"])["resale_price"].median().reset_index()
fig = px.line(monthly, x="month", y="resale_price", color="town", labels={"resale_price": "Median Price (SGD)", "month": "Month"})
st.plotly_chart(fig, use_container_width=True)

# MRT distance scatter
if filtered["nearest_mrt_dist_km"].notna().any():
    st.subheader("Median Price vs MRT Distance")
    town_mrt = filtered.groupby("town").agg(
        median_price=("resale_price", "median"),
        avg_mrt_dist=("nearest_mrt_dist_km", "mean"),
        count=("resale_price", "count"),
    ).reset_index()
    fig = px.scatter(
        town_mrt, x="avg_mrt_dist", y="median_price", size="count", text="town",
        labels={"avg_mrt_dist": "Avg MRT Distance (km)", "median_price": "Median Price (SGD)"},
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# Summary table
st.subheader("Summary Statistics")
summary = filtered.groupby("town").agg(
    median_price=("resale_price", "median"),
    median_psm=("price_per_sqm", "median"),
    avg_remaining_lease=("remaining_lease_years", "mean"),
    avg_mrt_dist=("nearest_mrt_dist_km", "mean"),
    transactions=("resale_price", "count"),
).round(1).sort_values("median_price", ascending=False).reset_index()

st.dataframe(summary, use_container_width=True)
```

**Step 2: Test manually by running Streamlit**

```bash
streamlit run app/Home.py
```

**Step 3: Commit**

```bash
git add app/pages/1_Town_Comparison.py
git commit -m "feat: add Town Comparison page with charts and filters"
```

---

### Task 11: Page 2 — Price Drivers

**Files:**
- Create: `app/pages/2_Price_Drivers.py`

**Step 1: Implement the Price Drivers page**

```python
# app/pages/2_Price_Drivers.py
"""Price Drivers — understand what affects HDB resale prices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import load_processed_data, load_prediction_models, get_flat_types

st.set_page_config(page_title="Price Drivers", layout="wide")
st.title("Price Drivers")

df = load_processed_data()
models = load_prediction_models()

# Filter
with st.sidebar:
    flat_type = st.selectbox("Flat Type", get_flat_types(df), index=get_flat_types(df).index("5 ROOM") if "5 ROOM" in get_flat_types(df) else 0)

filtered = df[df["flat_type"] == flat_type].copy()

# Feature importance
st.subheader("Feature Importance")
model = models["median"]
feature_names = models["feature_names"]

importance = model.feature_importances_
importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
importance_df = importance_df.sort_values("importance", ascending=True).tail(15)

fig = px.bar(importance_df, x="importance", y="feature", orientation="h",
             labels={"importance": "Importance", "feature": "Feature"})
st.plotly_chart(fig, use_container_width=True)

# Partial dependence plots
st.subheader("Partial Dependence")
st.markdown("See how each feature affects predicted price, holding other features constant.")

pdp_feature = st.selectbox(
    "Select feature",
    ["floor_area_sqm", "storey_mid", "remaining_lease_years", "nearest_mrt_dist_km"],
)

numeric_cols = ["floor_area_sqm", "storey_mid", "remaining_lease_years", "nearest_mrt_dist_km"]
clean = filtered.dropna(subset=numeric_cols + ["resale_price"])

if not clean.empty:
    from src.model.train import prepare_features

    X, y, fnames = prepare_features(clean)

    if pdp_feature in fnames:
        feat_idx = fnames.index(pdp_feature)
        feat_values = np.linspace(
            np.percentile(X[:, feat_idx], 5),
            np.percentile(X[:, feat_idx], 95),
            50,
        )

        # Calculate partial dependence
        mean_preds = []
        X_temp = X.copy()
        for val in feat_values:
            X_temp[:, feat_idx] = val
            mean_preds.append(model.predict(X_temp).mean())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feat_values, y=mean_preds, mode="lines"))
        fig.update_layout(
            xaxis_title=pdp_feature,
            yaxis_title="Average Predicted Price (SGD)",
        )
        st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
st.subheader("Correlation Matrix")
corr_cols = ["resale_price", "floor_area_sqm", "storey_mid", "remaining_lease_years", "nearest_mrt_dist_km", "price_per_sqm"]
existing_cols = [c for c in corr_cols if c in filtered.columns]
corr = filtered[existing_cols].corr()

fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
st.plotly_chart(fig, use_container_width=True)
```

**Step 2: Test manually**

```bash
streamlit run app/Home.py
```

**Step 3: Commit**

```bash
git add app/pages/2_Price_Drivers.py
git commit -m "feat: add Price Drivers page with feature importance and PDP"
```

---

### Task 12: Page 3 — Fair Price Estimator

**Files:**
- Create: `app/pages/3_Fair_Price_Estimator.py`

**Step 1: Implement the Fair Price Estimator page**

```python
# app/pages/3_Fair_Price_Estimator.py
"""Fair Price Estimator — estimate what a flat should cost."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime

import pandas as pd
import streamlit as st

from app.data_loader import load_processed_data, load_prediction_models, get_towns
from src.model.predict import predict_price, find_comparable_transactions

st.set_page_config(page_title="Fair Price Estimator", layout="wide")
st.title("Fair Price Estimator")

df = load_processed_data()
models = load_prediction_models()

# Input form
with st.form("estimator_form"):
    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", get_towns(df))
        floor_area = st.number_input("Floor Area (sqm)", min_value=50, max_value=200, value=110)
        storey = st.slider("Storey (midpoint)", 1, 50, 10)

    with col2:
        lease_commence = st.number_input(
            "Lease Commence Year", min_value=1966, max_value=datetime.now().year, value=1995
        )
        remaining_lease = 99 - (datetime.now().year - lease_commence)
        st.metric("Remaining Lease", f"{remaining_lease} years")
        mrt_dist = st.number_input("Distance to Nearest MRT (km)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    result = predict_price(
        models, town=town, floor_area=floor_area, storey=storey,
        remaining_lease=remaining_lease, mrt_dist=mrt_dist,
    )

    st.subheader("Estimated Fair Price")
    col1, col2, col3 = st.columns(3)
    col1.metric("Low (10th percentile)", f"${result.get('low', 0):,.0f}")
    col2.metric("Estimate (Median)", f"${result['estimate']:,.0f}")
    col3.metric("High (90th percentile)", f"${result.get('high', 0):,.0f}")

    # Comparable transactions
    st.subheader("Comparable Recent Transactions")
    comps = find_comparable_transactions(
        df, town=town, flat_type="5 ROOM", floor_area=floor_area, storey=storey, n=10,
    )
    if not comps.empty:
        display_cols = ["month", "block", "street_name", "storey_range", "floor_area_sqm",
                        "remaining_lease_years", "resale_price", "price_per_sqm"]
        existing = [c for c in display_cols if c in comps.columns]
        st.dataframe(comps[existing].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No comparable transactions found in the last 12 months.")
```

**Step 2: Test manually**

```bash
streamlit run app/Home.py
```

**Step 3: Commit**

```bash
git add app/pages/3_Fair_Price_Estimator.py
git commit -m "feat: add Fair Price Estimator page with confidence intervals"
```

---

### Task 13: Page 4 — Listing Evaluator

**Files:**
- Create: `app/pages/4_Listing_Evaluator.py`

**Step 1: Implement the Listing Evaluator page**

```python
# app/pages/4_Listing_Evaluator.py
"""Listing Evaluator — assess if a specific listing is fairly priced."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime

import plotly.express as px
import streamlit as st

from app.data_loader import load_processed_data, load_prediction_models, get_towns
from src.model.predict import predict_price, find_comparable_transactions

st.set_page_config(page_title="Listing Evaluator", layout="wide")
st.title("Listing Evaluator")
st.markdown("Enter details from a listing to see if it's fairly priced.")

df = load_processed_data()
models = load_prediction_models()

# Input form
with st.form("listing_form"):
    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", get_towns(df))
        asking_price = st.number_input("Asking Price (SGD)", min_value=100000, max_value=2000000, value=550000, step=10000)
        floor_area = st.number_input("Floor Area (sqm)", min_value=50, max_value=200, value=110)

    with col2:
        storey = st.slider("Storey (midpoint)", 1, 50, 10)
        lease_commence = st.number_input("Lease Commence Year", min_value=1966, max_value=datetime.now().year, value=1995)
        remaining_lease = 99 - (datetime.now().year - lease_commence)
        st.metric("Remaining Lease", f"{remaining_lease} years")
        mrt_dist = st.number_input("Distance to Nearest MRT (km)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    submitted = st.form_submit_button("Evaluate Listing")

if submitted:
    result = predict_price(
        models, town=town, floor_area=floor_area, storey=storey,
        remaining_lease=remaining_lease, mrt_dist=mrt_dist,
    )

    estimate = result["estimate"]
    diff = asking_price - estimate
    diff_pct = (diff / estimate) * 100

    # Verdict
    if diff_pct < -5:
        verdict = "Below Market"
        color = "green"
    elif diff_pct > 5:
        verdict = "Above Market"
        color = "red"
    else:
        verdict = "Fair"
        color = "orange"

    st.subheader("Valuation Assessment")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Asking Price", f"${asking_price:,.0f}")
    col2.metric("Model Estimate", f"${estimate:,.0f}")
    col3.metric("Difference", f"${diff:,.0f}", delta=f"{diff_pct:+.1f}%", delta_color="inverse")
    col4.markdown(f"### :{color}[{verdict}]")

    if "low" in result and "high" in result:
        st.caption(f"Model confidence range: ${result['low']:,.0f} – ${result['high']:,.0f}")

    # Price trend for this town + flat type
    st.subheader(f"Price Trend — {town} 5-ROOM")
    trend_data = df[
        (df["town"] == town) & (df["flat_type"] == "5 ROOM")
    ].groupby("month")["resale_price"].median().reset_index()

    if not trend_data.empty:
        # Show last 2 years
        trend_data = trend_data.sort_values("month").tail(24)
        fig = px.line(trend_data, x="month", y="resale_price",
                      labels={"resale_price": "Median Price (SGD)", "month": "Month"})
        fig.add_hline(y=asking_price, line_dash="dash", line_color="red",
                      annotation_text="Asking Price")
        fig.add_hline(y=estimate, line_dash="dash", line_color="green",
                      annotation_text="Model Estimate")
        st.plotly_chart(fig, use_container_width=True)

    # Comparable transactions
    st.subheader("Comparable Transactions (Last 12 Months)")
    comps = find_comparable_transactions(
        df, town=town, flat_type="5 ROOM", floor_area=floor_area, storey=storey, n=10,
    )
    if not comps.empty:
        display_cols = ["month", "block", "street_name", "storey_range", "floor_area_sqm",
                        "remaining_lease_years", "resale_price", "price_per_sqm"]
        existing = [c for c in display_cols if c in comps.columns]
        st.dataframe(comps[existing].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No comparable transactions found.")
```

**Step 2: Test manually**

```bash
streamlit run app/Home.py
```

**Step 3: Commit**

```bash
git add app/pages/4_Listing_Evaluator.py
git commit -m "feat: add Listing Evaluator page with verdict and comparables"
```

---

### Task 14: Data Pipeline Runner Script

**Files:**
- Create: `run_pipeline.py`

This script orchestrates the full pipeline: fetch → geocode → process → train.

**Step 1: Create the pipeline runner**

```python
# run_pipeline.py
"""Run the full data pipeline: fetch → geocode → process → train."""

import sys
sys.path.insert(0, ".")

from src.data.fetch import fetch_all_resale_data
from src.data.geocode import batch_geocode_blocks
from src.data.process import process_raw_data
from src.model.train import train_and_save
from src.utils.mrt import fetch_mrt_stations

import pandas as pd


def main():
    print("=" * 60)
    print("HDB Resale Valuation — Data Pipeline")
    print("=" * 60)

    # Step 1: Fetch raw data
    print("\n[1/5] Fetching resale data from data.gov.sg...")
    raw_csv = fetch_all_resale_data("data/raw")

    # Step 2: Fetch MRT stations
    print("\n[2/5] Fetching MRT station coordinates...")
    mrt_stations = fetch_mrt_stations("data/reference/mrt_stations.json")
    print(f"  → {len(mrt_stations)} MRT stations")

    # Step 3: Geocode HDB blocks
    print("\n[3/5] Geocoding HDB blocks (this may take a while on first run)...")
    raw_df = pd.read_csv(raw_csv)
    blocks = raw_df[["block", "street_name"]].drop_duplicates().to_dict("records")
    block_coords = batch_geocode_blocks(blocks, "data/reference/block_coords.json")
    print(f"  → {len(block_coords)} blocks geocoded")

    # Step 4: Process data
    print("\n[4/5] Processing and feature engineering...")
    processed_df = process_raw_data(
        raw_csv,
        block_coords=block_coords,
        mrt_stations=mrt_stations,
        output_path="data/processed/resale_processed.csv",
    )
    print(f"  → {len(processed_df)} processed records")

    # Step 5: Train model
    print("\n[5/5] Training XGBoost model...")
    # Use last 5 years for training
    recent = processed_df[processed_df["year"] >= processed_df["year"].max() - 5]
    train_and_save(recent, "models")

    print("\n" + "=" * 60)
    print("Pipeline complete! Run: streamlit run app/Home.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

**Step 2: Commit**

```bash
git add run_pipeline.py
git commit -m "feat: add data pipeline runner script"
```

---

### Task 15: End-to-End Test & Polish

**Step 1: Run the full pipeline**

```bash
source .venv/bin/activate
python run_pipeline.py
```

Expected: Data fetched, geocoded, processed, model trained. This will take several minutes on first run (geocoding is the bottleneck).

**Step 2: Launch the dashboard**

```bash
streamlit run app/Home.py
```

**Step 3: Verify each page works**

1. Home page loads with description
2. Town Comparison — filters work, charts render, table shows data
3. Price Drivers — feature importance chart, PDP works, correlation matrix renders
4. Fair Price Estimator — form submits, estimate shows with confidence interval, comparables appear
5. Listing Evaluator — form submits, verdict shows (Below/Fair/Above Market), trend chart with asking price line, comparables table

**Step 4: Fix any issues found during testing**

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: polish and finalize dashboard"
```
