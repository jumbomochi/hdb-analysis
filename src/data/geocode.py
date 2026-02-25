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


if __name__ == "__main__":
    # Quick test with a known address
    result = geocode_address("1 ANG MO KIO AVE 3")
    print(f"Result: {result}")
