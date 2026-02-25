"""Price prediction and comparable transaction lookup."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_models(models_dir: str = "models") -> dict:
    """Load all trained models.

    Expects model files saved by train.py:
      - xgb_median.joblib  (median regression model)
      - xgb_q0.1.joblib    (10th percentile quantile model)
      - xgb_q0.5.joblib    (50th percentile quantile model)
      - xgb_q0.9.joblib    (90th percentile quantile model)

    Returns a dict with keys: 'median', 'feature_names', 'q0.1', 'q0.5', 'q0.9'.
    """
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

    Parameters
    ----------
    models : dict
        Dictionary from load_models() containing trained models and feature names.
    town : str
        Town name (e.g. "TAMPINES").
    floor_area : float
        Floor area in square metres.
    storey : int
        Mid-point of storey range.
    remaining_lease : float
        Remaining lease in years.
    mrt_dist : float
        Distance to nearest MRT station in km.

    Returns
    -------
    dict
        Keys: 'estimate' (median prediction), 'low' (10th percentile),
        'high' (90th percentile). 'low' and 'high' are only present if
        quantile models were loaded.
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
    """Find the most similar recent transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Full resale transactions dataset with columns: town, flat_type,
        floor_area_sqm, storey_mid, remaining_lease_years, resale_price, month.
    town : str
        Target town.
    flat_type : str
        Target flat type (e.g. "4 ROOM").
    floor_area : float
        Target floor area in sqm.
    storey : int
        Target storey mid-point.
    n : int
        Maximum number of comparable transactions to return.
    months_back : int
        Only consider transactions from the last N months.

    Returns
    -------
    pd.DataFrame
        Up to n most similar transactions, sorted by similarity.
    """
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

    if filtered.empty:
        return filtered

    # Score by similarity (lower = more similar)
    filtered["_similarity"] = (
        abs(filtered["floor_area_sqm"] - floor_area) / 10
        + abs(filtered["storey_mid"] - storey) / 5
    )
    filtered = filtered.sort_values("_similarity").head(n)
    return filtered.drop(columns=["_similarity"])
