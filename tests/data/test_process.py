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
