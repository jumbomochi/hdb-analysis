# HDB Resale Flat Valuation Dashboard — Design

## Purpose

Interactive Streamlit dashboard to help a prospective 5-room HDB buyer understand resale flat valuations in Singapore. The tool answers: "Is this flat fairly priced?" by combining market analysis, price driver insights, and a machine learning price estimator.

## Data

### Source
- **HDB Resale Flat Prices** from data.gov.sg API (free, no auth). ~900k+ transactions from 2000–present.
- Fields: month, town, flat_type, block, street_name, storey_range, floor_area_sqm, flat_model, lease_commence_date, remaining_lease, resale_price.

### Enrichment
- **MRT proximity:** Geocode each HDB block via OneMap API, compute straight-line distance to nearest MRT station using a static reference list of station coordinates.

### Feature Engineering
- Parse `storey_range` into midpoint integer.
- Calculate `remaining_lease_years` from lease commence date.
- Derive `price_per_sqm`.
- Extract `year` and `quarter` from `month`.
- Calculate `nearest_mrt_distance_km`.

## Dashboard Pages

### Page 1: Town Comparison
- Filters: flat type (default 5-room), date range, town multi-select.
- Charts: median price by town (bar), price trend by town (line), price/sqm distribution (box plot), price vs MRT distance (scatter).
- Summary table: median price, median psm, avg remaining lease, avg MRT distance, transaction count per town.

### Page 2: Price Drivers
- Feature importance bar chart from trained model.
- Interactive partial dependence plots for key features (remaining lease, floor, MRT distance, floor area).
- Correlation heatmap of numeric features.

### Page 3: Fair Price Estimator
- Input form: town, storey range, floor area, lease commence date, nearest MRT distance.
- Output: estimated fair price with confidence interval (10th/50th/90th percentile), comparable recent transactions.

### Page 4: Listing Evaluator
- Input form: address, asking price, floor area, storey, lease commence date.
- Output: model estimate vs asking price (% difference), comparable transactions (last 6–12 months), town+type price trend, verdict badge (Below Market / Fair / Above Market).

## Technical Architecture

```
hdb-analysis/
├── data/
│   ├── raw/                    # Raw CSV from data.gov.sg
│   ├── processed/              # Cleaned + feature-engineered data
│   └── reference/              # MRT station coordinates
├── src/
│   ├── data/
│   │   ├── fetch.py            # Download from data.gov.sg API
│   │   ├── process.py          # Clean + feature engineering
│   │   └── geocode.py          # OneMap API for HDB block geocoding
│   ├── model/
│   │   ├── train.py            # Train XGBoost model
│   │   └── predict.py          # Inference + confidence intervals
│   └── utils/
│       └── mrt.py              # MRT distance calculations
├── app/
│   ├── Home.py                 # Streamlit entry point
│   └── pages/
│       ├── 1_Town_Comparison.py
│       ├── 2_Price_Drivers.py
│       ├── 3_Fair_Price_Estimator.py
│       └── 4_Listing_Evaluator.py
├── models/                     # Saved trained models
├── requirements.txt
└── README.md
```

## Model

- **Algorithm:** XGBoost regressor.
- **Training data:** Resale transactions from last 5 years (for current market relevance).
- **Features:** town (one-hot), floor_area_sqm, storey_midpoint, remaining_lease_years, nearest_mrt_distance_km.
- **Confidence intervals:** Quantile regression predicting 10th, 50th, 90th percentiles.
- **Retraining:** On data refresh.

## Key Dependencies

streamlit, pandas, plotly, scikit-learn, xgboost, requests
