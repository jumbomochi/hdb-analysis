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
