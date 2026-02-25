"""Shared styling for the HDB Resale Valuation Dashboard."""

import streamlit as st
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Color palette
# ---------------------------------------------------------------------------
EMERALD = "#10B981"
EMERALD_DARK = "#059669"
RED = "#EF4444"
AMBER = "#F59E0B"
SLATE_900 = "#0F172A"
SLATE_700 = "#334155"
SLATE_500 = "#64748B"
SLATE_200 = "#E2E8F0"
SLATE_100 = "#F1F5F9"
BG = "#F8FAFC"

COLORWAY = [EMERALD, "#3B82F6", AMBER, RED, "#8B5CF6", "#EC4899", "#06B6D4", "#F97316"]


def inject_custom_css() -> None:
    """Inject global CSS overrides for a polished SaaS feel."""
    st.markdown(
        f"""
        <style>
        /* Off-white page background */
        .stApp {{
            background-color: {BG};
        }}

        /* Metric cards */
        [data-testid="stMetric"] {{
            background: #FFFFFF;
            border: 1px solid {SLATE_200};
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        [data-testid="stMetricValue"] {{
            font-size: 1.8rem;
            font-weight: 700;
            color: {SLATE_900};
        }}
        [data-testid="stMetricLabel"] {{
            text-transform: uppercase;
            font-size: 0.75rem;
            letter-spacing: 0.05em;
            color: {SLATE_500};
        }}
        [data-testid="stMetricDelta"] > div {{
            font-size: 0.85rem;
        }}

        /* Form containers */
        [data-testid="stForm"] {{
            background: #FFFFFF;
            border: 1px solid {SLATE_200};
            border-radius: 12px;
            padding: 24px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}

        /* Dataframe containers */
        [data-testid="stDataFrame"] {{
            border: 1px solid {SLATE_200};
            border-radius: 8px;
            overflow: hidden;
        }}

        /* Tab styling */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
        }}
        .stTabs [data-baseweb="tab"] {{
            border-radius: 8px 8px 0 0;
            padding: 8px 20px;
            font-weight: 500;
        }}

        /* Sidebar */
        section[data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid {SLATE_200};
        }}

        /* Expander styling */
        [data-testid="stExpander"] {{
            border: 1px solid {SLATE_200};
            border-radius: 8px;
            background: #FFFFFF;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def apply_chart_style(fig: go.Figure) -> go.Figure:
    """Apply consistent Plotly styling to a figure."""
    fig.update_layout(
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", color=SLATE_700, size=13),
        colorway=COLORWAY,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=SLATE_200, gridwidth=1),
        yaxis=dict(gridcolor=SLATE_200, gridwidth=1),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=SLATE_200,
            font=dict(color=SLATE_900, size=13),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=SLATE_200,
            borderwidth=1,
            font=dict(size=12),
        ),
    )
    return fig


def render_price_map(df, height: int = 500) -> None:
    """Render a pydeck ScatterplotLayer map of HDB transactions.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: lat, lng, median_price, count.
        Typically the output of data_loader.get_map_data().
    height : int
        Map height in pixels.
    """
    import pydeck as pdk

    if df.empty or "lat" not in df.columns or "lng" not in df.columns:
        st.info("Map data not available.")
        return

    map_df = df.dropna(subset=["lat", "lng"]).copy()
    if map_df.empty:
        st.info("No geo-coded data available for the map.")
        return

    # Normalize price to 0-255 for color mapping (green = low, red = high)
    p_min = map_df["median_price"].min()
    p_max = map_df["median_price"].max()
    p_range = p_max - p_min if p_max != p_min else 1
    map_df["_norm"] = (map_df["median_price"] - p_min) / p_range
    map_df["r"] = (map_df["_norm"] * 239).astype(int).clip(0, 255)
    map_df["g"] = ((1 - map_df["_norm"]) * 185).astype(int).clip(0, 255)
    map_df["b"] = 69

    # Radius based on transaction count
    count_max = map_df["count"].max() if map_df["count"].max() > 0 else 1
    map_df["radius"] = 40 + (map_df["count"] / count_max) * 160

    layer = pdk.Layer(
        "ScatterplotLayer",
        data=map_df,
        get_position=["lng", "lat"],
        get_radius="radius",
        get_fill_color=["r", "g", "b", 180],
        pickable=True,
        auto_highlight=True,
    )

    tooltip = {
        "html": (
            "<b>{block}</b><br/>"
            "Median Price: <b>${median_price}</b><br/>"
            "Transactions: {count}"
        ),
        "style": {
            "backgroundColor": "#FFFFFF",
            "color": SLATE_900,
            "border": f"1px solid {SLATE_200}",
            "borderRadius": "8px",
            "padding": "8px 12px",
            "fontSize": "13px",
        },
    }

    view = pdk.ViewState(latitude=1.3521, longitude=103.8198, zoom=11, pitch=0)

    st.pydeck_chart(
        pdk.Deck(layers=[layer], initial_view_state=view, tooltip=tooltip),
        height=height,
    )
