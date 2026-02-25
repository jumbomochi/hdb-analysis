# app/pages/2_Price_Drivers.py
"""Price Drivers — understand what affects HDB resale prices."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data_loader import load_processed_data, load_prediction_models, get_flat_types
from app.styles import inject_custom_css, apply_chart_style, EMERALD, EMERALD_DARK

st.set_page_config(page_title="Price Drivers", layout="wide")
inject_custom_css()
st.title("Price Drivers")

df = load_processed_data()
models = load_prediction_models()

# Filter
with st.sidebar:
    flat_types = get_flat_types(df)
    flat_type = st.selectbox(
        "Flat Type",
        flat_types,
        index=flat_types.index("5 ROOM") if "5 ROOM" in flat_types else 0,
    )

filtered = df[df["flat_type"] == flat_type].copy()

model = models["median"]
feature_names = models["feature_names"]
importance = model.feature_importances_
importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
importance_df = importance_df.sort_values("importance", ascending=True).tail(15)

# ---------------------------------------------------------------------------
# KPI Row
# ---------------------------------------------------------------------------
k1, k2, k3 = st.columns(3)
k1.metric("Top Feature", importance_df.iloc[-1]["feature"].replace("_", " ").title())
k2.metric("Features Used", f"{len(feature_names)}")
k3.metric("Dataset Size", f"{len(filtered):,} rows")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_importance, tab_pdp, tab_corr = st.tabs(
    ["Feature Importance", "Partial Dependence", "Correlations"]
)

# --- Tab 1: Feature Importance ---
with tab_importance:
    st.subheader("Feature Importance")
    fig = px.bar(
        importance_df,
        x="importance",
        y="feature",
        orientation="h",
        labels={"importance": "Importance", "feature": "Feature"},
    )
    fig.update_traces(marker_color=EMERALD)
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)

# --- Tab 2: Partial Dependence ---
with tab_pdp:
    st.subheader("Partial Dependence")
    st.caption("See how each feature affects predicted price, holding other features constant.")

    pdp_feature = st.selectbox(
        "Select feature",
        ["floor_area_sqm", "storey_mid", "remaining_lease_years", "nearest_mrt_dist_km"],
    )

    numeric_cols = ["floor_area_sqm", "storey_mid", "remaining_lease_years", "nearest_mrt_dist_km"]
    clean = filtered.dropna(subset=numeric_cols + ["resale_price"])

    if not clean.empty:
        from src.model.train import prepare_features

        X, y, fnames = prepare_features(clean)

        if pdp_feature in fnames:
            feat_idx = fnames.index(pdp_feature)
            feat_values = np.linspace(
                np.percentile(X[:, feat_idx], 5),
                np.percentile(X[:, feat_idx], 95),
                50,
            )

            mean_preds = []
            X_temp = X.copy()
            for val in feat_values:
                X_temp[:, feat_idx] = val
                mean_preds.append(model.predict(X_temp).mean())

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=feat_values,
                    y=mean_preds,
                    mode="lines",
                    line=dict(color=EMERALD, width=3),
                    fill="tozeroy",
                    fillcolor="rgba(16, 185, 129, 0.1)",
                )
            )
            fig.update_layout(
                xaxis_title=pdp_feature.replace("_", " ").title(),
                yaxis_title="Average Predicted Price (SGD)",
            )
            apply_chart_style(fig)
            st.plotly_chart(fig, use_container_width=True)

# --- Tab 3: Correlations ---
with tab_corr:
    st.subheader("Correlation Matrix")
    corr_cols = [
        "resale_price", "floor_area_sqm", "storey_mid",
        "remaining_lease_years", "nearest_mrt_dist_km", "price_per_sqm",
    ]
    existing_cols = [c for c in corr_cols if c in filtered.columns]
    corr = filtered[existing_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale=[[0, "#EF4444"], [0.5, "#FFFFFF"], [1, EMERALD]],
        zmin=-1,
        zmax=1,
    )
    apply_chart_style(fig)
    st.plotly_chart(fig, use_container_width=True)
