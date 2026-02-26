# app/pages/4_Listing_Evaluator.py
"""Listing Evaluator — assess if a specific listing is fairly priced."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from datetime import datetime

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import load_processed_data, load_prediction_models, get_towns, get_flat_types
from app.styles import inject_custom_css, apply_chart_style, EMERALD, RED, AMBER, SLATE_500, SLATE_200, SLATE_900
from src.model.predict import predict_price, find_comparable_transactions

st.set_page_config(page_title="Listing Evaluator", layout="wide")
inject_custom_css()
st.title("Listing Evaluator")
st.caption("Enter details from a listing to see if it's fairly priced.")

df = load_processed_data()
models = load_prediction_models()

# Input form
with st.form("listing_form"):
    col1, col2 = st.columns(2)

    with col1:
        town = st.selectbox("Town", get_towns(df))
        flat_types = get_flat_types(df)
        flat_type = st.selectbox(
            "Flat Type",
            flat_types,
            index=flat_types.index("5 ROOM") if "5 ROOM" in flat_types else 0,
        )
        asking_price = st.number_input(
            "Asking Price (SGD)", min_value=100000, max_value=2000000, value=550000, step=10000
        )

    with col2:
        floor_area = st.number_input("Floor Area (sqm)", min_value=50, max_value=200, value=110)
        storey = st.slider("Storey (midpoint)", 1, 50, 10)
        lease_commence = st.number_input(
            "Lease Commence Year", min_value=1966, max_value=datetime.now().year, value=1995
        )
        remaining_lease = 99 - (datetime.now().year - lease_commence)
        st.metric("Remaining Lease", f"{remaining_lease} years")
        mrt_dist = st.number_input(
            "Distance to Nearest MRT (km)", min_value=0.0, max_value=5.0, value=0.5, step=0.1
        )

    submitted = st.form_submit_button("Evaluate Listing")

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
    diff = asking_price - estimate
    diff_pct = (diff / estimate) * 100

    # Verdict
    if diff_pct < -5:
        verdict = "Below Market — Potential Bargain"
        bg_color = "rgba(107, 127, 58, 0.08)"
        border_color = EMERALD
        text_color = EMERALD
    elif diff_pct > 5:
        verdict = "Above Market — Overpriced"
        bg_color = "rgba(194, 65, 12, 0.08)"
        border_color = RED
        text_color = RED
    else:
        verdict = "Fairly Priced"
        bg_color = "rgba(217, 119, 6, 0.08)"
        border_color = AMBER
        text_color = AMBER

    # Verdict banner
    st.markdown(
        f"""
        <div style="
            background: {bg_color};
            border-left: 5px solid {border_color};
            border-radius: 0 12px 12px 0;
            padding: clamp(12px, 3vw, 20px) clamp(14px, 3vw, 24px);
            margin: 8px 0 20px 0;
        ">
            <div style="font-size: clamp(1rem, 3vw, 1.4rem); font-weight: 700; color: {text_color};">
                {verdict}
            </div>
            <div style="font-size: 0.95rem; color: {SLATE_500}; margin-top: 4px;">
                Asking ${asking_price:,.0f} vs estimate ${estimate:,.0f}
                &mdash; {diff_pct:+.1f}% ({'+' if diff > 0 else ''}{diff:,.0f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics row
    c1, c2, c3 = st.columns(3)
    c1.metric("Asking Price", f"${asking_price:,.0f}")
    c2.metric("Model Estimate", f"${estimate:,.0f}")
    c3.metric("Difference", f"${diff:,.0f}", delta=f"{diff_pct:+.1f}%", delta_color="inverse")

    if "low" in result and "high" in result:
        st.caption(f"Model confidence range: ${result['low']:,.0f} – ${result['high']:,.0f}")

    # Tabs
    tab_trend, tab_comps, tab_details = st.tabs(
        ["Price Trend", "Comparable Transactions", "Valuation Details"]
    )

    with tab_trend:
        st.subheader(f"Price Trend — {town} {flat_type}")
        trend_data = (
            df[(df["town"] == town) & (df["flat_type"] == flat_type)]
            .groupby("month")["resale_price"]
            .median()
            .reset_index()
        )

        if not trend_data.empty:
            trend_data = trend_data.sort_values("month").tail(24)

            fig = go.Figure()

            # Shade gap between asking and estimate
            y_top = max(asking_price, estimate)
            y_bot = min(asking_price, estimate)
            fig.add_hrect(
                y0=y_bot,
                y1=y_top,
                fillcolor="rgba(100,100,100,0.06)",
                line_width=0,
                annotation_text=f"Gap: ${abs(diff):,.0f}",
                annotation_position="top right",
                annotation_font_size=11,
                annotation_font_color=SLATE_500,
            )

            fig.add_trace(
                go.Scatter(
                    x=trend_data["month"],
                    y=trend_data["resale_price"],
                    mode="lines",
                    name="Median Price",
                    line=dict(color=EMERALD, width=2.5),
                )
            )

            fig.add_hline(
                y=asking_price, line_dash="dash", line_color=RED,
                annotation_text="Asking Price", annotation_font_color=RED,
            )
            fig.add_hline(
                y=estimate, line_dash="dash", line_color=EMERALD,
                annotation_text="Model Estimate", annotation_font_color=EMERALD,
            )

            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Median Price (SGD)",
            )
            apply_chart_style(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info(f"No trend data for {town} — {flat_type}.")

    with tab_comps:
        st.subheader("Comparable Transactions (Last 12 Months)")
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
                file_name="listing_comparables.csv",
                mime="text/csv",
            )
        else:
            st.info("No comparable transactions found.")

    with tab_details:
        st.subheader("Valuation Details")
        st.markdown(f"""
        | Parameter | Value |
        |-----------|-------|
        | Town | {town} |
        | Flat Type | {flat_type} |
        | Floor Area | {floor_area} sqm |
        | Storey | {storey} |
        | Remaining Lease | {remaining_lease} years |
        | MRT Distance | {mrt_dist} km |
        | **Model Estimate** | **${estimate:,.0f}** |
        | **Asking Price** | **${asking_price:,.0f}** |
        | **Difference** | **{diff_pct:+.1f}%** |
        """)
