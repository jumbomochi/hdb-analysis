# app/data_loader.py
"""Shared data loading for Streamlit pages."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model.predict import load_models


@st.cache_data
def load_processed_data() -> pd.DataFrame:
    """Load the processed resale data."""
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / "resale_processed.csv"
    if not path.exists():
        st.error(f"Processed data not found at {path}. Run the data pipeline first.")
        st.stop()
    return pd.read_csv(path)


@st.cache_resource
def load_prediction_models() -> dict:
    """Load trained models."""
    models_dir = Path(__file__).resolve().parent.parent / "models"
    models = load_models(str(models_dir))
    if "median" not in models:
        st.error("Trained model not found. Run the training pipeline first.")
        st.stop()
    return models


def get_towns(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique towns."""
    return sorted(df["town"].unique().tolist())


def get_flat_types(df: pd.DataFrame) -> list[str]:
    """Get sorted list of unique flat types."""
    return sorted(df["flat_type"].unique().tolist())


@st.cache_data
def get_map_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions to block level for map display.

    Returns a DataFrame with columns: lat, lng, block, median_price, median_psm, count.
    """
    geo = df.dropna(subset=["lat", "lng"]).copy()
    if geo.empty:
        return pd.DataFrame()

    geo["lat_r"] = geo["lat"].round(5)
    geo["lng_r"] = geo["lng"].round(5)

    agg = (
        geo.groupby(["lat_r", "lng_r"])
        .agg(
            lat=("lat", "first"),
            lng=("lng", "first"),
            block=("block", "first"),
            median_price=("resale_price", "median"),
            median_psm=("price_per_sqm", "median"),
            count=("resale_price", "count"),
        )
        .reset_index(drop=True)
    )
    agg["median_price"] = agg["median_price"].round(0).astype(int)
    agg["median_psm"] = agg["median_psm"].round(0).astype(int)
    return agg


@st.cache_data
def get_kpi_data(df: pd.DataFrame) -> dict:
    """Compute dashboard-wide KPIs."""
    median_price = df["resale_price"].median()
    total_transactions = len(df)
    most_popular_town = df["town"].value_counts().idxmax()
    median_psm = df["price_per_sqm"].median()

    max_year = int(df["year"].max())
    prev_year = max_year - 1
    curr = df[df["year"] == max_year]["resale_price"].median()
    prev = df[df["year"] == prev_year]["resale_price"].median()
    yoy_change_pct = ((curr - prev) / prev * 100) if prev else 0.0

    return {
        "median_price": median_price,
        "yoy_change_pct": round(yoy_change_pct, 1),
        "total_transactions": total_transactions,
        "most_popular_town": most_popular_town,
        "median_psm": median_psm,
    }
