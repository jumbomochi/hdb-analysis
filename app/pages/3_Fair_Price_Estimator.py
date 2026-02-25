# app/pages/3_Fair_Price_Estimator.py
"""Fair Price Estimator — estimate what a flat should cost."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime

import pandas as pd
import streamlit as st

from app.data_loader import load_processed_data, load_prediction_models, get_towns, get_flat_types
from app.styles import inject_custom_css, EMERALD, SLATE_500, SLATE_200, SLATE_900
from src.model.predict import predict_price, find_comparable_transactions

st.set_page_config(page_title="Fair Price Estimator", layout="wide")
inject_custom_css()
st.title("Fair Price Estimator")

df = load_processed_data()
models = load_prediction_models()

# Input form
with st.form("estimator_form"):
    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", get_towns(df))
        flat_types = get_flat_types(df)
        flat_type = st.selectbox(
            "Flat Type",
            flat_types,
            index=flat_types.index("5 ROOM") if "5 ROOM" in flat_types else 0,
        )
        floor_area = st.number_input("Floor Area (sqm)", min_value=50, max_value=200, value=110)

    with col2:
        storey = st.slider("Storey (midpoint)", 1, 50, 10)
        lease_commence = st.number_input(
            "Lease Commence Year", min_value=1966, max_value=datetime.now().year, value=1995
        )
        remaining_lease = 99 - (datetime.now().year - lease_commence)
        st.metric("Remaining Lease", f"{remaining_lease} years")
        mrt_dist = st.number_input(
            "Distance to Nearest MRT (km)", min_value=0.0, max_value=5.0, value=0.5, step=0.1
        )

    submitted = st.form_submit_button("Estimate Price")

if submitted:
    result = predict_price(
        models,
        town=town,
        floor_area=floor_area,
        storey=storey,
        remaining_lease=remaining_lease,
        mrt_dist=mrt_dist,
    )

    estimate = result["estimate"]
    low = result.get("low", 0)
    high = result.get("high", 0)

    # Hero estimate card
    st.markdown(
        f"""
        <div style="
            background: #FFFFFF;
            border: 2px solid {EMERALD};
            border-radius: 16px;
            padding: 32px;
            text-align: center;
            margin: 16px 0 24px 0;
            box-shadow: 0 2px 8px rgba(16, 185, 129, 0.12);
        ">
            <div style="font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; color: {SLATE_500}; margin-bottom: 4px;">
                Estimated Fair Price
            </div>
            <div style="font-size: 2.8rem; font-weight: 800; color: {EMERALD};">
                ${estimate:,.0f}
            </div>
            <div style="font-size: 0.9rem; color: {SLATE_500}; margin-top: 4px;">
                Confidence range: ${low:,.0f} &ndash; ${high:,.0f}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Tabs
    tab_breakdown, tab_comps = st.tabs(["Price Breakdown", "Comparable Transactions"])

    with tab_breakdown:
        st.subheader("Comparison to Town Median")
        town_data = df[(df["town"] == town) & (df["flat_type"] == flat_type)]
        if not town_data.empty:
            town_median = town_data["resale_price"].median()
            diff = estimate - town_median
            diff_pct = (diff / town_median) * 100

            c1, c2, c3 = st.columns(3)
            c1.metric("Model Estimate", f"${estimate:,.0f}")
            c2.metric("Town Median", f"${town_median:,.0f}")
            c3.metric("Difference", f"${diff:,.0f}", delta=f"{diff_pct:+.1f}%")
        else:
            st.info(f"No data for {town} — {flat_type}.")

    with tab_comps:
        st.subheader("Comparable Recent Transactions")
        comps = find_comparable_transactions(
            df, town=town, flat_type=flat_type, floor_area=floor_area, storey=storey, n=10,
        )
        if not comps.empty:
            display_cols = [
                "month", "block", "street_name", "storey_range",
                "floor_area_sqm", "remaining_lease_years", "resale_price", "price_per_sqm",
            ]
            existing = [c for c in display_cols if c in comps.columns]
            st.dataframe(comps[existing].reset_index(drop=True), use_container_width=True)

            csv = comps[existing].to_csv(index=False)
            st.download_button(
                "Download Comparables CSV",
                data=csv,
                file_name="comparable_transactions.csv",
                mime="text/csv",
            )
        else:
            st.info("No comparable transactions found in the last 12 months.")
