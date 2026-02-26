# app/pages/1_Town_Comparison.py
"""Town Comparison — compare resale prices across towns."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data_loader import load_processed_data, get_towns, get_flat_types, get_map_data
from app.styles import inject_custom_css, apply_chart_style, render_price_map, BLUE

st.set_page_config(page_title="Town Comparison", layout="wide")
inject_custom_css()
st.title("Town Comparison")

df = load_processed_data()

# ---------------------------------------------------------------------------
# Sidebar filters — grouped with expanders
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Filters")

    with st.expander("Location Filters", expanded=True):
        all_towns = get_towns(df)
        selected_towns = st.multiselect("Towns (leave empty for all)", all_towns)

    with st.expander("Property Filters", expanded=True):
        flat_types = get_flat_types(df)
        flat_type = st.selectbox(
            "Flat Type",
            flat_types,
            index=flat_types.index("5 ROOM") if "5 ROOM" in flat_types else 0,
        )
        year_range = st.slider(
            "Year Range",
            int(df["year"].min()),
            int(df["year"].max()),
            (int(df["year"].max()) - 3, int(df["year"].max())),
        )

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

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4)
k1.metric("Median Price", f"${filtered['resale_price'].median():,.0f}")
k2.metric("Median Price/sqm", f"${filtered['price_per_sqm'].median():,.0f}")
k3.metric("Transactions", f"{len(filtered):,}")
k4.metric("Towns", f"{filtered['town'].nunique()}")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_trends, tab_map, tab_data = st.tabs(
    ["Price Overview", "Trends & Analysis", "Map View", "Data Table"]
)

# --- Tab 1: Price Overview ---
with tab_overview:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Median Resale Price by Town")
        town_median = (
            filtered.groupby("town")["resale_price"]
            .median()
            .sort_values(ascending=False)
            .reset_index()
        )
        fig = px.bar(
            town_median,
            x="town",
            y="resale_price",
            labels={"resale_price": "Median Price (SGD)", "town": "Town"},
        )
        fig.update_traces(
            marker_color=BLUE,
            text=town_median["resale_price"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside",
            textfont_size=10,
        )
        fig.update_layout(xaxis_tickangle=-45)
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Price per sqm Distribution")
        fig = px.box(
            filtered,
            x="town",
            y="price_per_sqm",
            labels={"price_per_sqm": "Price/sqm (SGD)", "town": "Town"},
        )
        fig.update_layout(xaxis_tickangle=-45)
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Trends & Analysis ---
with tab_trends:
    st.subheader("Price Trend Over Time")
    monthly = filtered.groupby(["month", "town"])["resale_price"].median().reset_index()
    fig = px.line(
        monthly,
        x="month",
        y="resale_price",
        color="town",
        labels={"resale_price": "Median Price (SGD)", "month": "Month"},
    )
    # Annotate latest point per town
    for town_name in monthly["town"].unique():
        town_data = monthly[monthly["town"] == town_name].sort_values("month")
        if not town_data.empty:
            latest = town_data.iloc[-1]
            fig.add_annotation(
                x=latest["month"],
                y=latest["resale_price"],
                text=f"${latest['resale_price']:,.0f}",
                showarrow=False,
                font=dict(size=10),
                yshift=12,
            )
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)

    if filtered["nearest_mrt_dist_km"].notna().any():
        st.subheader("Median Price vs MRT Distance")
        town_mrt = filtered.groupby("town").agg(
            median_price=("resale_price", "median"),
            avg_mrt_dist=("nearest_mrt_dist_km", "mean"),
            count=("resale_price", "count"),
        ).reset_index()
        fig = px.scatter(
            town_mrt,
            x="avg_mrt_dist",
            y="median_price",
            size="count",
            text="town",
            labels={
                "avg_mrt_dist": "Avg MRT Distance (km)",
                "median_price": "Median Price (SGD)",
            },
        )
        fig.update_traces(textposition="top center")
        apply_chart_style(fig)
        st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Map View ---
with tab_map:
    st.subheader("Transaction Map")
    st.caption("Blocks color-coded by median price — blue (lower) to gold (higher)")
    map_df = get_map_data(filtered)
    render_price_map(map_df, height=550)

# --- Tab 4: Data Table ---
with tab_data:
    st.subheader("Summary Statistics")
    summary = (
        filtered.groupby("town")
        .agg(
            median_price=("resale_price", "median"),
            median_psm=("price_per_sqm", "median"),
            avg_remaining_lease=("remaining_lease_years", "mean"),
            avg_mrt_dist=("nearest_mrt_dist_km", "mean"),
            transactions=("resale_price", "count"),
        )
        .round(1)
        .sort_values("median_price", ascending=False)
        .reset_index()
    )
    st.dataframe(summary, use_container_width=True)

    # Export button
    csv = summary.to_csv(index=False)
    st.download_button(
        "Download CSV",
        data=csv,
        file_name="town_comparison.csv",
        mime="text/csv",
    )
