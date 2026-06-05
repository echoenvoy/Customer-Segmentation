"""
features.py
===========
Phase 3 - Feature Engineering

Aggregates the cleaned transaction table to one row per customer and computes
seven behavioral features:

  Feature            Formula
  -------------------------------------------------------------------------
  Recency            Days between the customer's last purchase and the
                     reference date (max InvoiceDate in the dataset + 1 day)
  Frequency          Number of unique invoices (orders)
  Monetary           Sum of Revenue (Quantity × UnitPrice)
  AvgOrderValue      Monetary / Frequency
  UniqueProducts     Number of distinct StockCode values purchased
  ActiveDays         Days between first and last purchase
  CancellationRate   Cancelled invoices ÷ total invoices (from raw data)

The output is saved to data/processed/customer_features.csv.

Run: python src/features.py
"""

import os
import pandas as pd

# -- Paths ----------------------------------------------------------------------
RAW_CSV      = "data/raw/online_retail_raw.csv"       # needed for cancellations
CLEAN_CSV    = "data/processed/transactions_clean.csv"
FEATURES_CSV = "data/processed/customer_features.csv"


def load_data():
    """Load both the clean transactions and the raw data (for cancellation rates)."""
    clean = pd.read_csv(CLEAN_CSV, dtype={"CustomerID": str, "InvoiceNo": str})
    clean["InvoiceDate"] = pd.to_datetime(clean["InvoiceDate"])

    raw = pd.read_csv(RAW_CSV, dtype={"CustomerID": str, "InvoiceNo": str})
    raw["InvoiceDate"] = pd.to_datetime(raw["InvoiceDate"])
    # Keep only rows that have a CustomerID (same filter as clean_data.py)
    raw = raw.dropna(subset=["CustomerID"])
    return clean, raw


def compute_rfm(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Recency, Frequency, Monetary value.

    Reference date = last invoice date in the dataset + 1 day.
    This is a standard choice so the most recent customer gets Recency = 1.
    """
    reference_date = clean["InvoiceDate"].max() + pd.Timedelta(days=1)

    rfm = (
        clean.groupby("CustomerID")
        .agg(
            LastPurchase=("InvoiceDate", "max"),
            Frequency    =("InvoiceNo",   "nunique"),   # unique orders
            Monetary     =("Revenue",     "sum"),
        )
        .reset_index()
    )

    # Recency in whole days
    rfm["Recency"] = (reference_date - rfm["LastPurchase"]).dt.days
    rfm = rfm.drop(columns=["LastPurchase"])
    return rfm


def compute_extra_features(clean: pd.DataFrame) -> pd.DataFrame:
    """
    Compute AvgOrderValue, UniqueProducts, and ActiveDays.

      AvgOrderValue  - separates frequent small buyers from infrequent big spenders
      UniqueProducts - measures how diverse a customer's shopping basket is
      ActiveDays     - 0 for one-time buyers; higher = long-term relationship
    """
    extra = (
        clean.groupby("CustomerID")
        .agg(
            TotalRevenue    =("Revenue",    "sum"),
            TotalOrders     =("InvoiceNo",  "nunique"),
            UniqueProducts  =("StockCode",  "nunique"),
            FirstPurchase   =("InvoiceDate","min"),
            LastPurchase    =("InvoiceDate","max"),
        )
        .reset_index()
    )

    extra["AvgOrderValue"] = extra["TotalRevenue"] / extra["TotalOrders"]
    extra["ActiveDays"]    = (extra["LastPurchase"] - extra["FirstPurchase"]).dt.days

    # Keep only the derived columns (RFM handles Revenue & Frequency)
    extra = extra[["CustomerID", "AvgOrderValue", "UniqueProducts", "ActiveDays"]]
    return extra


def compute_cancellation_rate(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the fraction of invoices that were cancellations per customer.

    CancellationRate = cancelled_invoices / total_invoices
    A high rate may indicate problem customers or heavy returners.
    """
    # Flag each invoice-row as cancelled or not
    raw["IsCancelled"] = raw["InvoiceNo"].str.startswith("C")

    cancel = (
        raw.groupby("CustomerID")
        .agg(
            TotalInvoices      =("InvoiceNo",    "nunique"),
            CancelledInvoices  =("IsCancelled",  "sum"),   # sum of True = count
        )
        .reset_index()
    )

    cancel["CancellationRate"] = (
        cancel["CancelledInvoices"] / cancel["TotalInvoices"]
    ).fillna(0)

    return cancel[["CustomerID", "CancellationRate"]]


def build_features(clean: pd.DataFrame, raw: pd.DataFrame) -> pd.DataFrame:
    """Merge all feature groups into a single customer-level table."""
    rfm    = compute_rfm(clean)
    extra  = compute_extra_features(clean)
    cancel = compute_cancellation_rate(raw)

    # Left-join so every clean customer appears (not all may have cancellations)
    features = (
        rfm
        .merge(extra,  on="CustomerID", how="left")
        .merge(cancel, on="CustomerID", how="left")
    )

    # Fill NaN cancellation rate with 0 (customer never cancelled)
    features["CancellationRate"] = features["CancellationRate"].fillna(0)

    return features


if __name__ == "__main__":
    clean, raw = load_data()
    features   = build_features(clean, raw)

    os.makedirs("data/processed", exist_ok=True)
    features.to_csv(FEATURES_CSV, index=False)

    print(f"[INFO] Customer feature table saved to {FEATURES_CSV}")
    print(f"[INFO] Shape: {features.shape}")
    print("\n[Feature statistics]")
    print(features.describe().round(2).to_string())
