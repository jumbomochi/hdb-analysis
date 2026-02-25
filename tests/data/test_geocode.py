import tempfile

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

    with tempfile.TemporaryDirectory() as tmpdir:
        cache_path = f"{tmpdir}/block_coords.json"
        with patch("src.data.geocode.geocode_address", side_effect=mock_geocode):
            results = batch_geocode_blocks(blocks, cache_path=cache_path, delay=0)

    assert call_count == 2  # Only 2 unique addresses
    assert len(results) == 2
