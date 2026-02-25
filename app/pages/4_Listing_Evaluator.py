# app/pages/4_Listing_Evaluator.py
"""Listing Evaluator — assess if a specific listing is fairly priced."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime

import plotly.express as px
import streamlit as st

from app.data_loader import load_processed_data, load_prediction_models, get_towns
from src.model.predict import predict_price, find_comparable_transactions

st.set_page_config(page_title="Listing Evaluator", layout="wide")
st.title("Listing Evaluator")
st.markdown("Enter details from a listing to see if it's fairly priced.")

df = load_processed_data()
models = load_prediction_models()

# Input form
with st.form("listing_form"):
    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", get_towns(df))
        asking_price = st.number_input("Asking Price (SGD)", min_value=100000, max_value=2000000, value=550000, step=10000)
        floor_area = st.number_input("Floor Area (sqm)", min_value=50, max_value=200, value=110)

    with col2:
        storey = st.slider("Storey (midpoint)", 1, 50, 10)
        lease_commence = st.number_input("Lease Commence Year", min_value=1966, max_value=datetime.now().year, value=1995)
        remaining_lease = 99 - (datetime.now().year - lease_commence)
        st.metric("Remaining Lease", f"{remaining_lease} years")
        mrt_dist = st.number_input("Distance to Nearest MRT (km)", min_value=0.0, max_value=5.0, value=0.5, step=0.1)

    submitted = st.form_submit_button("Evaluate Listing")

if submitted:
    result = predict_price(
        models, town=town, floor_area=floor_area, storey=storey,
        remaining_lease=remaining_lease, mrt_dist=mrt_dist,
    )

    estimate = result["estimate"]
    diff = asking_price - estimate
    diff_pct = (diff / estimate) * 100

    # Verdict
    if diff_pct < -5:
        verdict = "Below Market"
        color = "green"
    elif diff_pct > 5:
        verdict = "Above Market"
        color = "red"
    else:
        verdict = "Fair"
        color = "orange"

    st.subheader("Valuation Assessment")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Asking Price", f"${asking_price:,.0f}")
    col2.metric("Model Estimate", f"${estimate:,.0f}")
    col3.metric("Difference", f"${diff:,.0f}", delta=f"{diff_pct:+.1f}%", delta_color="inverse")
    col4.markdown(f"### :{color}[{verdict}]")

    if "low" in result and "high" in result:
        st.caption(f"Model confidence range: ${result['low']:,.0f} – ${result['high']:,.0f}")

    # Price trend for this town + flat type
    st.subheader(f"Price Trend — {town} 5-ROOM")
    trend_data = df[
        (df["town"] == town) & (df["flat_type"] == "5 ROOM")
    ].groupby("month")["resale_price"].median().reset_index()

    if not trend_data.empty:
        # Show last 2 years
        trend_data = trend_data.sort_values("month").tail(24)
        fig = px.line(trend_data, x="month", y="resale_price",
                      labels={"resale_price": "Median Price (SGD)", "month": "Month"})
        fig.add_hline(y=asking_price, line_dash="dash", line_color="red",
                      annotation_text="Asking Price")
        fig.add_hline(y=estimate, line_dash="dash", line_color="green",
                      annotation_text="Model Estimate")
        st.plotly_chart(fig, use_container_width=True)

    # Comparable transactions
    st.subheader("Comparable Transactions (Last 12 Months)")
    comps = find_comparable_transactions(
        df, town=town, flat_type="5 ROOM", floor_area=floor_area, storey=storey, n=10,
    )
    if not comps.empty:
        display_cols = ["month", "block", "street_name", "storey_range", "floor_area_sqm",
                        "remaining_lease_years", "resale_price", "price_per_sqm"]
        existing = [c for c in display_cols if c in comps.columns]
        st.dataframe(comps[existing].reset_index(drop=True), use_container_width=True)
    else:
        st.info("No comparable transactions found.")
