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
