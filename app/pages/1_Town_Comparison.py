# app/pages/1_Town_Comparison.py
"""Town Comparison — compare resale prices across towns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data_loader import load_processed_data, get_towns, get_flat_types

st.set_page_config(page_title="Town Comparison", layout="wide")
st.title("Town Comparison")

df = load_processed_data()

# Sidebar filters
with st.sidebar:
    st.header("Filters")
    flat_type = st.selectbox("Flat Type", get_flat_types(df), index=get_flat_types(df).index("5 ROOM") if "5 ROOM" in get_flat_types(df) else 0)
    year_range = st.slider(
        "Year Range",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].max()) - 3, int(df["year"].max())),
    )
    all_towns = get_towns(df)
    selected_towns = st.multiselect("Towns (leave empty for all)", all_towns)

# Filter data
filtered = df[
    (df["flat_type"] == flat_type)
    & (df["year"] >= year_range[0])
    & (df["year"] <= year_range[1])
]
if selected_towns:
    filtered = filtered[filtered["town"].isin(selected_towns)]

if filtered.empty:
    st.warning("No data matching filters.")
    st.stop()

# Median price by town
col1, col2 = st.columns(2)

with col1:
    st.subheader("Median Resale Price by Town")
    town_median = filtered.groupby("town")["resale_price"].median().sort_values(ascending=False).reset_index()
    fig = px.bar(town_median, x="town", y="resale_price", labels={"resale_price": "Median Price (SGD)", "town": "Town"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Price per sqm Distribution by Town")
    fig = px.box(filtered, x="town", y="price_per_sqm", labels={"price_per_sqm": "Price/sqm (SGD)", "town": "Town"})
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

# Price trend
st.subheader("Price Trend Over Time")
monthly = filtered.groupby(["month", "town"])["resale_price"].median().reset_index()
fig = px.line(monthly, x="month", y="resale_price", color="town", labels={"resale_price": "Median Price (SGD)", "month": "Month"})
st.plotly_chart(fig, use_container_width=True)

# MRT distance scatter
if filtered["nearest_mrt_dist_km"].notna().any():
    st.subheader("Median Price vs MRT Distance")
    town_mrt = filtered.groupby("town").agg(
        median_price=("resale_price", "median"),
        avg_mrt_dist=("nearest_mrt_dist_km", "mean"),
        count=("resale_price", "count"),
    ).reset_index()
    fig = px.scatter(
        town_mrt, x="avg_mrt_dist", y="median_price", size="count", text="town",
        labels={"avg_mrt_dist": "Avg MRT Distance (km)", "median_price": "Median Price (SGD)"},
    )
    fig.update_traces(textposition="top center")
    st.plotly_chart(fig, use_container_width=True)

# Summary table
st.subheader("Summary Statistics")
summary = filtered.groupby("town").agg(
    median_price=("resale_price", "median"),
    median_psm=("price_per_sqm", "median"),
    avg_remaining_lease=("remaining_lease_years", "mean"),
    avg_mrt_dist=("nearest_mrt_dist_km", "mean"),
    transactions=("resale_price", "count"),
).round(1).sort_values("median_price", ascending=False).reset_index()

st.dataframe(summary, use_container_width=True)
