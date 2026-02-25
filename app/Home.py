# app/Home.py
"""HDB Resale Flat Valuation Dashboard — Home page."""

import streamlit as st

st.set_page_config(
    page_title="HDB Resale Valuation",
    page_icon="🏠",
    layout="wide",
)

st.title("HDB Resale Flat Valuation Dashboard")
st.markdown(
    """
    An interactive tool to help you evaluate HDB resale flat prices in Singapore.

    ### Pages

    - **Town Comparison** — Compare median prices, trends, and value across towns
    - **Price Drivers** — Understand which factors impact resale prices the most
    - **Fair Price Estimator** — Get an estimated fair price for a flat based on its attributes
    - **Listing Evaluator** — Evaluate a specific listing against the model and recent transactions

    ### Data

    Resale transaction data from [data.gov.sg](https://data.gov.sg/collections/189/view),
    enriched with MRT proximity from OneMap.

    Use the sidebar to navigate between pages.
    """
)
