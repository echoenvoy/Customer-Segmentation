
import argparse
import os
import sys

import pandas as pd

# Add the src folder to the path so we can import scripts directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


RAW_CSV = "data/raw/online_retail_raw.csv"
REQUIRED_COLUMNS = [
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the customer segmentation pipeline.")
    parser.add_argument(
        "--input",
        help=(
            "Optional CSV or Excel file to use as raw transactions. "
            "If omitted, the UCI Online Retail dataset is used."
        ),
    )
    return parser.parse_args()


def validate_input_columns(df):
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            "Input data is missing required columns: " + ", ".join(missing)
        )


def load_custom_input(path):
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, dtype={"CustomerID": str, "InvoiceNo": str})
    elif ext == ".csv":
        df = pd.read_csv(path, dtype={"CustomerID": str, "InvoiceNo": str})
    else:
        raise ValueError("Custom input must be a .csv, .xlsx, or .xls file.")

    validate_input_columns(df)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df[REQUIRED_COLUMNS]


args = parse_args()

# -- Phase 1: Load & describe ---------------------------------------------------
print("\n" + "=" * 60)
print("  PHASE 1 - DATA ACQUISITION & DESCRIPTION")
print("=" * 60)

import load_describe_data

if args.input:
    print(f"[INFO] Using custom input file: {args.input}")
    df_raw = load_custom_input(args.input)
else:
    load_describe_data.download_dataset()
    df_raw = load_describe_data.load_data(load_describe_data.RAW_FILE)

load_describe_data.audit(df_raw)
os.makedirs("data/raw", exist_ok=True)
df_raw.to_csv(RAW_CSV, index=False)
print(f"[INFO] Raw CSV saved: {RAW_CSV}")

# -- Phase 2: Clean -------------------------------------------------------------
print("\n" + "=" * 60)
print("  PHASE 2 - DATA CLEANING")
print("=" * 60)

import clean_data
raw_df   = clean_data.load_raw(clean_data.RAW_CSV)
clean_df = clean_data.clean(raw_df)
os.makedirs("data/processed", exist_ok=True)
clean_df.to_csv(clean_data.CLEAN_CSV, index=False)
print(f"[INFO] Clean CSV saved: {clean_data.CLEAN_CSV}")

# -- Phase 3: Feature engineering ----------------------------------------------
print("\n" + "=" * 60)
print("  PHASE 3 - FEATURE ENGINEERING")
print("=" * 60)

import features as feat
clean_loaded = pd.read_csv(feat.CLEAN_CSV, dtype={"CustomerID": str})
clean_loaded["InvoiceDate"] = pd.to_datetime(clean_loaded["InvoiceDate"])
raw_loaded = pd.read_csv(feat.RAW_CSV, dtype={"CustomerID": str})
raw_loaded["InvoiceDate"] = pd.to_datetime(raw_loaded["InvoiceDate"])
raw_loaded = raw_loaded.dropna(subset=["CustomerID"])

feature_df = feat.build_features(clean_loaded, raw_loaded)
feature_df.to_csv(feat.FEATURES_CSV, index=False)
print(f"[INFO] Features saved: {feat.FEATURES_CSV}")

# -- Phase 3b: EDA -------------------------------------------------------------
print("\n" + "=" * 60)
print("  PHASE 3b - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

import eda
df_clean  = pd.read_csv(eda.CLEAN_CSV, dtype={"CustomerID": str})
df_clean["InvoiceDate"] = pd.to_datetime(df_clean["InvoiceDate"])
df_feat   = pd.read_csv(eda.FEATURES_CSV, dtype={"CustomerID": str})

eda.plot_revenue_by_country(df_clean)
eda.plot_monthly_revenue(df_clean.copy())
eda.plot_orders_by_weekday(df_clean.copy())
eda.plot_quantity_distribution(df_clean)
eda.plot_price_distribution(df_clean)
eda.plot_top_products(df_clean)
eda.plot_recency_distribution(df_feat)
eda.plot_rfm_pairplot(df_feat)
print(f"[INFO] EDA figures saved to {eda.FIG_DIR}/")

# -- Phase 3c: SQL storage -----------------------------------------------------
print("\n" + "=" * 60)
print("  PHASE 3c - SQL STORAGE & QUERIES")
print("=" * 60)

import store_sql
store_sql.load_to_sqlite(df_clean, df_feat)
store_sql.run_all_queries()

# -- Phase 4: Clustering --------------------------------------------------------
print("\n" + "=" * 60)
print("  PHASE 4 - CLUSTERING & EVALUATION")
print("=" * 60)

# Run the clustering script as __main__ by exec
with open("src/train_cluster_models.py") as f:
    exec(f.read())

print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print("=" * 60)
print("""
Output summary
--------------
  data/raw/online_retail_raw.csv           - raw transactions (CSV)
  data/processed/transactions_clean.csv    - cleaned transactions
  data/processed/customer_features.csv     - customer feature table
  data/processed/customers_clustered.csv   - customers with cluster labels
  data/processed/retail.db                 - SQLite database
  models/best_clustering_pipeline.joblib   - trained scaler + KMeans pipeline
  reports/eda_figures/                     - 10 EDA and evaluation figures
""")
