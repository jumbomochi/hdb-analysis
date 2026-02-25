import pytest
from unittest.mock import patch, MagicMock
from src.utils.mrt import (
    fetch_mrt_stations,
    haversine_distance,
    find_nearest_mrt,
)


def test_haversine_distance_known_values():
    """Test haversine with known Singapore distances."""
    # Tampines MRT to Pasir Ris MRT ~ 2.5 km
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
