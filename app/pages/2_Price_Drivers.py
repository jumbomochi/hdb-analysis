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

st.set_page_config(page_title="Price Drivers", layout="wide")
st.title("Price Drivers")

df = load_processed_data()
models = load_prediction_models()

# Filter
with st.sidebar:
    flat_type = st.selectbox("Flat Type", get_flat_types(df), index=get_flat_types(df).index("5 ROOM") if "5 ROOM" in get_flat_types(df) else 0)

filtered = df[df["flat_type"] == flat_type].copy()

# Feature importance
st.subheader("Feature Importance")
model = models["median"]
feature_names = models["feature_names"]

importance = model.feature_importances_
importance_df = pd.DataFrame({"feature": feature_names, "importance": importance})
importance_df = importance_df.sort_values("importance", ascending=True).tail(15)

fig = px.bar(importance_df, x="importance", y="feature", orientation="h",
             labels={"importance": "Importance", "feature": "Feature"})
st.plotly_chart(fig, use_container_width=True)

# Partial dependence plots
st.subheader("Partial Dependence")
st.markdown("See how each feature affects predicted price, holding other features constant.")

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

        # Calculate partial dependence
        mean_preds = []
        X_temp = X.copy()
        for val in feat_values:
            X_temp[:, feat_idx] = val
            mean_preds.append(model.predict(X_temp).mean())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=feat_values, y=mean_preds, mode="lines"))
        fig.update_layout(
            xaxis_title=pdp_feature,
            yaxis_title="Average Predicted Price (SGD)",
        )
        st.plotly_chart(fig, use_container_width=True)

# Correlation heatmap
st.subheader("Correlation Matrix")
corr_cols = ["resale_price", "floor_area_sqm", "storey_mid", "remaining_lease_years", "nearest_mrt_dist_km", "price_per_sqm"]
existing_cols = [c for c in corr_cols if c in filtered.columns]
corr = filtered[existing_cols].corr()

fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
st.plotly_chart(fig, use_container_width=True)
