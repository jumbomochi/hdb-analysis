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
    if remaining_str and isinstance(remaining_str, str):
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
        df["address_key"] = df["block"].astype(str) + " " + df["street_name"]
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
