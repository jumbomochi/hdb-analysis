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
