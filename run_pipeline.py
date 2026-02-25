# run_pipeline.py
"""Run the full data pipeline: fetch -> geocode -> process -> train."""

import sys
sys.path.insert(0, ".")

from src.data.fetch import fetch_all_resale_data
from src.data.geocode import batch_geocode_blocks
from src.data.process import process_raw_data
from src.model.train import train_and_save
from src.utils.mrt import fetch_mrt_stations

import pandas as pd


def main():
    print("=" * 60)
    print("HDB Resale Valuation — Data Pipeline")
    print("=" * 60)

    # Step 1: Fetch raw data
    print("\n[1/5] Fetching resale data from data.gov.sg...")
    raw_csv = fetch_all_resale_data("data/raw")

    # Step 2: Fetch MRT stations
    print("\n[2/5] Fetching MRT station coordinates...")
    mrt_stations = fetch_mrt_stations("data/reference/mrt_stations.json")
    print(f"  -> {len(mrt_stations)} MRT stations")

    # Step 3: Geocode HDB blocks
    print("\n[3/5] Geocoding HDB blocks (this may take a while on first run)...")
    raw_df = pd.read_csv(raw_csv)
    blocks = raw_df[["block", "street_name"]].drop_duplicates().to_dict("records")
    block_coords = batch_geocode_blocks(blocks, "data/reference/block_coords.json")
    print(f"  -> {len(block_coords)} blocks geocoded")

    # Step 4: Process data
    print("\n[4/5] Processing and feature engineering...")
    processed_df = process_raw_data(
        raw_csv,
        block_coords=block_coords,
        mrt_stations=mrt_stations,
        output_path="data/processed/resale_processed.csv",
    )
    print(f"  -> {len(processed_df)} processed records")

    # Step 5: Train model
    print("\n[5/5] Training XGBoost model...")
    # Use last 5 years for training
    recent = processed_df[processed_df["year"] >= processed_df["year"].max() - 5]
    train_and_save(recent, "models")

    print("\n" + "=" * 60)
    print("Pipeline complete! Run: streamlit run app/Home.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
