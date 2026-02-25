"""Train XGBoost model for HDB resale price prediction."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


def prepare_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare feature matrix and target from processed dataframe.

    Returns (X, y, feature_names).
    """
    feature_cols = ["floor_area_sqm", "storey_mid", "remaining_lease_years", "nearest_mrt_dist_km"]
    df_clean = df.dropna(subset=feature_cols + ["resale_price"]).copy()

    # One-hot encode town
    town_dummies = pd.get_dummies(df_clean["town"], prefix="town", dtype=float)
    numeric = df_clean[feature_cols].astype(float)
    X_df = pd.concat([numeric, town_dummies], axis=1)

    feature_names = list(X_df.columns)
    X = X_df.values
    y = df_clean["resale_price"].values.astype(float)

    return X, y, feature_names


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int = 500,
    max_depth: int = 6,
    learning_rate: float = 0.05,
) -> XGBRegressor:
    """Train an XGBoost regressor."""
    model = XGBRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    return model


def train_quantile_models(
    X: np.ndarray,
    y: np.ndarray,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> dict[float, XGBRegressor]:
    """Train separate models for each quantile (for confidence intervals)."""
    models = {}
    for q in quantiles:
        model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            objective="reg:quantileerror",
            quantile_alpha=q,
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X, y)
        models[q] = model
    return models


def save_model(model, feature_names: list[str], path: str = "models/xgb_model.joblib"):
    """Save model and feature names."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names}, path)


def train_and_save(df: pd.DataFrame, models_dir: str = "models"):
    """Full training pipeline: prepare, train median + quantile models, save."""
    X, y, feature_names = prepare_features(df)

    print(f"Training on {len(y)} samples with {len(feature_names)} features...")

    # Train median model
    median_model = train_model(X, y)
    save_model(median_model, feature_names, f"{models_dir}/xgb_median.joblib")

    # Train quantile models for confidence intervals
    quantile_models = train_quantile_models(X, y)
    for q, model in quantile_models.items():
        save_model(model, feature_names, f"{models_dir}/xgb_q{q}.joblib")

    # Evaluate
    from sklearn.metrics import r2_score, mean_absolute_error
    preds = median_model.predict(X)
    print(f"Training R²: {r2_score(y, preds):.4f}")
    print(f"Training MAE: ${mean_absolute_error(y, preds):,.0f}")

    return median_model, feature_names
