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
