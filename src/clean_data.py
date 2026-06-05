"""
clean_data.py
=============
Phase 2 - Data Wrangling

Applies a series of cleaning steps to the raw Online Retail dataset and saves
a cleaned transaction table to data/processed/transactions_clean.csv.

Cleaning steps (each step is logged so you can trace the row counts):
  1. Remove exact duplicate rows
  2. Drop rows with missing CustomerID  (can't cluster without an ID)
  3. Remove cancelled invoices (InvoiceNo starts with 'C')
  4. Remove rows with Quantity  <= 0
  5. Remove rows with UnitPrice <= 0
  6. Add a Revenue column  (Quantity × UnitPrice)

Run: python src/clean_data.py
"""

import os
import pandas as pd

# -- Paths ----------------------------------------------------------------------
RAW_CSV      = "data/raw/online_retail_raw.csv"
CLEAN_CSV    = "data/processed/transactions_clean.csv"
PROCESSED_DIR = "data/processed"


def load_raw(path: str) -> pd.DataFrame:
    """Load the raw CSV produced by load_describe_data.py."""
    df = pd.read_csv(path, dtype={"CustomerID": str, "InvoiceNo": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


def log_step(label: str, before: int, after: int):
    """Print a one-line summary of how many rows were removed in a step."""
    removed = before - after
    print(f"  [{label}]  {before:>8,} -> {after:>8,}   (removed {removed:,})")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Run all cleaning steps and return the cleaned DataFrame."""

    print("\n-- Cleaning pipeline --------------------------------------")
    n0 = len(df)
    print(f"  [Start]          {n0:>8,} rows")

    # Step 1 - Remove exact duplicate rows
    df = df.drop_duplicates()
    log_step("Drop duplicates", n0, len(df)); n0 = len(df)

    # Step 2 - Drop rows without a CustomerID
    # These rows are unusable for customer-level clustering
    df = df.dropna(subset=["CustomerID"])
    log_step("Drop missing CustomerID", n0, len(df)); n0 = len(df)

    # Step 3 - Remove cancelled invoices (prefix 'C')
    # Cancellations are not real purchases; we track them separately via the
    # cancellation-rate feature in features.py
    mask_cancel = df["InvoiceNo"].str.startswith("C")
    df = df[~mask_cancel]
    log_step("Remove cancellations", n0, len(df)); n0 = len(df)

    # Step 4 - Remove rows with non-positive Quantity
    df = df[df["Quantity"] > 0]
    log_step("Remove Quantity <= 0", n0, len(df)); n0 = len(df)

    # Step 5 - Remove rows with non-positive UnitPrice
    df = df[df["UnitPrice"] > 0]
    log_step("Remove UnitPrice <= 0", n0, len(df)); n0 = len(df)

    # Step 6 - Add Revenue column (used heavily in feature engineering)
    df["Revenue"] = df["Quantity"] * df["UnitPrice"]

    print(f"  [Final]          {len(df):>8,} rows")
    print("-----------------------------------------------------------\n")

    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    raw = load_raw(RAW_CSV)
    clean_df = clean(raw)

    clean_df.to_csv(CLEAN_CSV, index=False)
    print(f"[INFO] Clean transactions saved to {CLEAN_CSV}")
    print(f"[INFO] Unique customers after cleaning: {clean_df['CustomerID'].nunique():,}")
