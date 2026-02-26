# app/Home.py
"""HDB Resale Flat Valuation Dashboard — Home page."""

import streamlit as st

from app.data_loader import load_processed_data, get_map_data, get_kpi_data
from app.styles import inject_custom_css, render_price_map, BLUE, SLATE_500, SLATE_200, CARD_BG

st.set_page_config(
    page_title="HDB Resale Valuation",
    page_icon="🏠",
    layout="wide",
)

inject_custom_css()

st.title("HDB Resale Flat Valuation Dashboard")
st.caption("An interactive tool to evaluate HDB resale flat prices in Singapore")

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
df = load_processed_data()
kpis = get_kpi_data(df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Median Resale Price", f"${kpis['median_price']:,.0f}", f"{kpis['yoy_change_pct']:+.1f}% YoY")
k2.metric("Total Transactions", f"{kpis['total_transactions']:,}")
k3.metric("Most Active Town", kpis["most_popular_town"])
k4.metric("Median Price / sqm", f"${kpis['median_psm']:,.0f}")

# ---------------------------------------------------------------------------
# Hero Map
# ---------------------------------------------------------------------------
st.markdown("### Transaction Map")
st.caption("Each dot represents a block — color indicates median price (blue = lower, gold = higher)")

map_data = get_map_data(df)
render_price_map(map_data, height=520)

# ---------------------------------------------------------------------------
# Navigation Cards
# ---------------------------------------------------------------------------
st.markdown("### Explore the Dashboard")

NAV_CARDS = [
    {
        "title": "Town Comparison",
        "desc": "Compare median prices, trends, and value across towns with interactive charts and maps.",
        "icon": "📊",
    },
    {
        "title": "Price Drivers",
        "desc": "Understand which factors — floor area, storey, lease, MRT distance — impact prices the most.",
        "icon": "🔍",
    },
    {
        "title": "Fair Price Estimator",
        "desc": "Get a model-estimated fair price for any flat based on its attributes and location.",
        "icon": "💰",
    },
    {
        "title": "Listing Evaluator",
        "desc": "Paste a listing's details to see if it's priced above, below, or at market value.",
        "icon": "✅",
    },
]

cols = st.columns(4)
for col, card in zip(cols, NAV_CARDS):
    col.markdown(
        f"""
        <div style="
            background: {CARD_BG};
            border: 1px solid {SLATE_200};
            border-radius: 12px;
            padding: clamp(14px, 3vw, 24px) clamp(12px, 2.5vw, 20px);
            min-height: auto;
            margin-bottom: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.04);
        ">
            <div style="font-size: clamp(1.3rem, 4vw, 1.8rem); margin-bottom: 8px;">{card['icon']}</div>
            <div style="font-weight: 700; font-size: 1rem; margin-bottom: 6px; color: {BLUE};">
                {card['title']}
            </div>
            <div style="font-size: 0.85rem; color: {SLATE_500}; line-height: 1.4;">
                {card['desc']}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")
st.caption(
    "Data from [data.gov.sg](https://data.gov.sg/collections/189/view), "
    "enriched with MRT proximity from OneMap. Use the sidebar to navigate."
)
