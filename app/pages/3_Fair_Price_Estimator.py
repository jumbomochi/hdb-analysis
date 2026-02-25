# app/pages/3_Fair_Price_Estimator.py
"""Fair Price Estimator — estimate what a flat should cost."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime

import pandas as pd
import streamlit as st

from app.data_loader import load_processed_data, load_prediction_models, get_towns
from src.model.predict import predict_price, find_comparable_transactions

st.set_page_config(page_title="Fair Price Estimator", layout="wide")
st.title("Fair Price Estimator")

df = load_processed_data()
models = load_prediction_models()

# Input form
with st.form("estimator_form"):
    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", get_towns(df))
        floor_area = st.number_input("Floor Area (sqm)", min_value=50, max_value=200, value=110)
        storey = st.slider("Storey (midpoint)", 1, 50, 10)

    with col2:
        lease_commence = st.number_input(
            "Lease Commence Year", min_value=1966, max_value=datetime.now().year, value=1995
        )
        remaining_lease = 99 - (datetime.now().year - lease_commence)
        st.metric("Remaining Lease", f"{remaining_lease} years")
        mrt_dist = st.number_input("Distance to Nearest MRT (km)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    result = predict_price(
        models, town=town, floor_area=floor_area, storey=storey,
        remaining_lease=remaining_lease, mrt_dist=mrt_dist,
    )

    st.subheader("Estimated Fair Price")
    col1, col2, col3 = st.columns(3)
    col1.metric("Low (10th percentile)", f"${result.get('low', 0):,.0f}")
    col2.metric("Estimate (Median)", f"${result['estimate']:,.0f}")
    col3.metric("High (90th percentile)", f"${result.get('high', 0):,.0f}")

    # Comparable transactions
    st.subheader("Comparable Recent Transactions")
    comps = find_comparable_transactions(
        df, town=town, flat_type="5 ROOM", floor_area=floor_area, storey=storey, n=10,
    )
    if not comps.empty:
        display_cols = ["month", "block", "street_name", "storey_range", "floor_area_sqm",
                        "remaining_lease_years", "resale_price", "price_per_sqm"]
        existing = [c for c in display_cols if c in comps.columns]
        st.dataframe(comps[existing].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No comparable transactions found in the last 12 months.")
