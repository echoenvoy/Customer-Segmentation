"""
load_describe_data.py
=====================
Phase 1 - Data Acquisition & Initial Audit

This script downloads the UCI Online Retail dataset (if not already present),
loads it into a pandas DataFrame, and prints a full audit report:
shape, dtypes, missing values, duplicates, and descriptive statistics.

Run: python src/load_describe_data.py
"""

import os
import urllib.request
import pandas as pd

# -- Paths ----------------------------------------------------------------------
RAW_DIR  = "data/raw"
RAW_FILE = os.path.join(RAW_DIR, "OnlineRetail.xlsx")

# UCI direct download URL (Excel version)
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
)

# -- Download helper ------------------------------------------------------------
def download_dataset():
    """Download the dataset zip from UCI and extract the Excel file."""
    import zipfile, io

    os.makedirs(RAW_DIR, exist_ok=True)

    if os.path.exists(RAW_FILE):
        print(f"[INFO] Dataset already present at {RAW_FILE}. Skipping download.")
        return

    print("[INFO] Downloading dataset from UCI …")
    response = urllib.request.urlopen(DATA_URL)
    zipped   = zipfile.ZipFile(io.BytesIO(response.read()))

    # Find the Excel file inside the zip
    excel_name = [n for n in zipped.namelist() if n.endswith(".xlsx")][0]
    zipped.extract(excel_name, RAW_DIR)

    # Rename to a consistent filename
    extracted_path = os.path.join(RAW_DIR, excel_name)
    os.rename(extracted_path, RAW_FILE)
    print(f"[INFO] Saved to {RAW_FILE}")


# -- Load -----------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    """Read the Excel file and return a raw DataFrame."""
    print(f"[INFO] Loading {path} …")
    df = pd.read_excel(path, dtype={"CustomerID": str})
    # Parse InvoiceDate as datetime right away
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


# -- Audit report ---------------------------------------------------------------
def audit(df: pd.DataFrame):
    """Print a structured data quality report to stdout."""

    sep = "-" * 60

    print(f"\n{sep}")
    print("  DATASET AUDIT REPORT")
    print(sep)

    # 1. Shape
    print(f"\n[Shape]  {df.shape[0]:,} rows  ×  {df.shape[1]} columns")

    # 2. Column data types
    print(f"\n[Data types]")
    print(df.dtypes.to_string())

    # 3. Missing values (absolute + percentage)
    print(f"\n[Missing values]")
    missing = df.isna().sum()
    pct     = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({"count": missing, "pct_%": pct})
    print(summary[summary["count"] > 0].to_string())

    # 4. Duplicate rows
    n_dup = df.duplicated().sum()
    print(f"\n[Duplicate rows]  {n_dup:,}")

    # 5. Cancelled invoices (InvoiceNo starts with 'C')
    n_cancel = df["InvoiceNo"].astype(str).str.startswith("C").sum()
    print(f"\n[Cancelled invoices (prefix 'C')]  {n_cancel:,}")

    # 6. Negative / zero Quantity and UnitPrice
    print(f"\n[Quantity <= 0]    {(df['Quantity']   <= 0).sum():,}")
    print(f"[UnitPrice <= 0]   {(df['UnitPrice']  <= 0).sum():,}")

    # 7. Descriptive statistics
    print(f"\n[Descriptive statistics - numerical columns]")
    print(df.describe().to_string())

    # 8. Unique customers / products / countries
    print(f"\n[Unique CustomerID (raw)]  {df['CustomerID'].nunique():,}")
    print(f"[Unique StockCode]          {df['StockCode'].nunique():,}")
    print(f"[Unique Country]            {df['Country'].nunique():,}")
    print(f"[Date range]  {df['InvoiceDate'].min().date()}  ->  {df['InvoiceDate'].max().date()}")

    print(f"\n{sep}\n")


# -- Main -----------------------------------------------------------------------
if __name__ == "__main__":
    download_dataset()
    df = load_data(RAW_FILE)
    audit(df)
    # Save the raw dataframe as CSV for faster loading in later scripts
    csv_path = os.path.join(RAW_DIR, "online_retail_raw.csv")
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Raw CSV saved to {csv_path}")
