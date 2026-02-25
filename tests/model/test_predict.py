import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock
from src.model.predict import predict_price, find_comparable_transactions


def _recent_month():
    """Return current month string for test data that needs to pass the months_back filter."""
    return pd.Timestamp.now().strftime("%Y-%m")


def test_find_comparable_transactions():
    recent = _recent_month()
    df = pd.DataFrame({
        "town": ["TAMPINES"] * 5 + ["BEDOK"] * 3,
        "flat_type": ["5 ROOM"] * 8,
        "floor_area_sqm": [110, 112, 108, 115, 105, 110, 112, 108],
        "storey_mid": [8, 10, 5, 15, 3, 8, 10, 12],
        "remaining_lease_years": [65, 63, 70, 55, 80, 65, 63, 70],
        "resale_price": [500000, 510000, 480000, 530000, 470000, 490000, 500000, 480000],
        "month": [recent] * 8,
    })

    comps = find_comparable_transactions(
        df, town="TAMPINES", flat_type="5 ROOM", floor_area=110, storey=8, n=3
    )

    assert 1 <= len(comps) <= 3
    assert all(comps["town"] == "TAMPINES")


def test_find_comparable_transactions_empty_when_no_match():
    """Should return empty DataFrame when no matching town/flat_type."""
    df = pd.DataFrame({
        "town": ["BEDOK"] * 3,
        "flat_type": ["3 ROOM"] * 3,
        "floor_area_sqm": [68, 70, 72],
        "storey_mid": [5, 8, 10],
        "remaining_lease_years": [60, 65, 70],
        "resale_price": [300000, 310000, 320000],
        "month": ["2024-01"] * 3,
    })

    comps = find_comparable_transactions(
        df, town="TAMPINES", flat_type="5 ROOM", floor_area=110, storey=8
    )
    assert len(comps) == 0


def test_find_comparable_transactions_similarity_sorting():
    """Closer matches in floor area and storey should rank higher."""
    recent = _recent_month()
    df = pd.DataFrame({
        "town": ["TAMPINES"] * 4,
        "flat_type": ["4 ROOM"] * 4,
        "floor_area_sqm": [90, 110, 92, 130],  # 110 is closest to target 108
        "storey_mid": [8, 8, 8, 8],
        "remaining_lease_years": [65, 65, 65, 65],
        "resale_price": [400000, 500000, 410000, 600000],
        "month": [recent] * 4,
    })

    comps = find_comparable_transactions(
        df, town="TAMPINES", flat_type="4 ROOM", floor_area=108, storey=8, n=2
    )
    assert len(comps) == 2
    # The two closest by floor area to 108 should be 110 (diff=2) and 90 (diff=18)
    # Actually 92 (diff=16) is closer than 90 (diff=18)
    areas = comps["floor_area_sqm"].tolist()
    assert 110 in areas  # closest match


def test_find_comparable_no_similarity_column_in_output():
    """Internal _similarity column should not appear in output."""
    recent = _recent_month()
    df = pd.DataFrame({
        "town": ["TAMPINES"] * 3,
        "flat_type": ["4 ROOM"] * 3,
        "floor_area_sqm": [90, 110, 92],
        "storey_mid": [8, 8, 8],
        "remaining_lease_years": [65, 65, 65],
        "resale_price": [400000, 500000, 410000],
        "month": [recent] * 3,
    })

    comps = find_comparable_transactions(
        df, town="TAMPINES", flat_type="4 ROOM", floor_area=108, storey=8
    )
    assert "_similarity" not in comps.columns


def test_predict_price_returns_estimate():
    """predict_price should return a dict with at least an 'estimate' key."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([550000.0])

    models = {
        "median": mock_model,
        "feature_names": ["floor_area_sqm", "storey_mid", "remaining_lease_years",
                          "nearest_mrt_dist_km", "town_TAMPINES", "town_BEDOK"],
    }

    result = predict_price(
        models,
        town="TAMPINES",
        floor_area=110.0,
        storey=8,
        remaining_lease=65.0,
        mrt_dist=0.5,
    )

    assert "estimate" in result
    assert result["estimate"] == pytest.approx(550000.0)


def test_predict_price_with_quantiles():
    """predict_price should return low/high when quantile models are present."""
    mock_median = MagicMock()
    mock_median.predict.return_value = np.array([550000.0])
    mock_q01 = MagicMock()
    mock_q01.predict.return_value = np.array([480000.0])
    mock_q09 = MagicMock()
    mock_q09.predict.return_value = np.array([620000.0])

    models = {
        "median": mock_median,
        "q0.1": mock_q01,
        "q0.9": mock_q09,
        "feature_names": ["floor_area_sqm", "storey_mid", "remaining_lease_years",
                          "nearest_mrt_dist_km", "town_TAMPINES", "town_BEDOK"],
    }

    result = predict_price(
        models,
        town="TAMPINES",
        floor_area=110.0,
        storey=8,
        remaining_lease=65.0,
        mrt_dist=0.5,
    )

    assert result["estimate"] == pytest.approx(550000.0)
    assert result["low"] == pytest.approx(480000.0)
    assert result["high"] == pytest.approx(620000.0)


def test_predict_price_correct_feature_vector():
    """Verify that the feature vector is constructed correctly with one-hot encoding."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([500000.0])

    feature_names = ["floor_area_sqm", "storey_mid", "remaining_lease_years",
                     "nearest_mrt_dist_km", "town_BEDOK", "town_TAMPINES", "town_WOODLANDS"]

    models = {
        "median": mock_model,
        "feature_names": feature_names,
    }

    predict_price(
        models,
        town="TAMPINES",
        floor_area=100.0,
        storey=10,
        remaining_lease=70.0,
        mrt_dist=0.8,
    )

    # Check the feature array passed to predict
    call_args = mock_model.predict.call_args
    X = call_args[0][0]
    assert X.shape == (1, 7)
    assert X[0, 0] == pytest.approx(100.0)  # floor_area_sqm
    assert X[0, 1] == pytest.approx(10.0)   # storey_mid
    assert X[0, 2] == pytest.approx(70.0)   # remaining_lease_years
    assert X[0, 3] == pytest.approx(0.8)    # nearest_mrt_dist_km
    assert X[0, 4] == pytest.approx(0.0)    # town_BEDOK = 0
    assert X[0, 5] == pytest.approx(1.0)    # town_TAMPINES = 1
    assert X[0, 6] == pytest.approx(0.0)    # town_WOODLANDS = 0
