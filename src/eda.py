"""
eda.py
======
Phase 2 - Exploratory Data Analysis

Produces 8 visualizations saved to reports/eda_figures/:

  1. Revenue by Country (top 10)
  2. Monthly Revenue trend
  3. Orders per day of week
  4. Distribution of Quantity (log scale)
  5. Distribution of UnitPrice (log scale)
  6. Top 10 best-selling products (by revenue)
  7. Customer Recency distribution
  8. RFM pairplot (Recency, Frequency, Monetary)

Run after clean_data.py and features.py have been executed.
Run: python src/eda.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# -- Paths ----------------------------------------------------------------------
CLEAN_CSV    = "data/processed/transactions_clean.csv"
FEATURES_CSV = "data/processed/customer_features.csv"
FIG_DIR      = "reports/eda_figures"

# -- Style ----------------------------------------------------------------------
sns.set_theme(style="whitegrid", palette="muted")
FIGSIZE = (9, 5)


def save(fig, name: str):
    """Save a figure and close it to free memory."""
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# -- Plot functions -------------------------------------------------------------

def plot_revenue_by_country(df):
    """Bar chart: top 10 countries by total revenue (excluding UK as outlier)."""
    rev = (
        df.groupby("Country")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .iloc[1:11]          # skip rank-1 (United Kingdom dominates)
    )
    fig, ax = plt.subplots(figsize=FIGSIZE)
    sns.barplot(x=rev.values, y=rev.index, ax=ax, color="steelblue")
    ax.set_title("Top 10 Countries by Revenue (excl. UK)", fontsize=13)
    ax.set_xlabel("Total Revenue (£)")
    ax.set_ylabel("")
    save(fig, "01_revenue_by_country.png")


def plot_monthly_revenue(df):
    """Line chart: total revenue aggregated by month."""
    df["YearMonth"] = df["InvoiceDate"].dt.to_period("M")
    monthly = df.groupby("YearMonth")["Revenue"].sum().reset_index()
    monthly["YearMonth"] = monthly["YearMonth"].astype(str)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(monthly["YearMonth"], monthly["Revenue"], marker="o", color="teal")
    ax.set_title("Monthly Revenue Trend", fontsize=13)
    ax.set_xlabel("Month")
    ax.set_ylabel("Revenue (£)")
    plt.xticks(rotation=45, ha="right")
    save(fig, "02_monthly_revenue.png")


def plot_orders_by_weekday(df):
    """Bar chart: number of invoices per day of the week."""
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    df["Weekday"] = df["InvoiceDate"].dt.dayofweek
    orders = df.groupby("Weekday")["InvoiceNo"].nunique()

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar([day_names[d] for d in orders.index], orders.values, color="coral")
    ax.set_title("Number of Orders by Day of Week", fontsize=13)
    ax.set_xlabel("Day")
    ax.set_ylabel("Unique Invoices")
    save(fig, "03_orders_by_weekday.png")


def plot_quantity_distribution(df):
    """Histogram of Quantity on a log scale to handle extreme outliers."""
    # Filter data to visual limits before binning to prevent outliers from compressing bins
    qty_filtered = df["Quantity"][df["Quantity"] <= 200]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(qty_filtered, bins=80, color="mediumpurple", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Distribution of Quantity (log y-axis)", fontsize=13)
    ax.set_xlabel("Quantity per Line")
    ax.set_ylabel("Frequency (log)")
    ax.set_xlim(0, 200)
    save(fig, "04_quantity_distribution.png")


def plot_price_distribution(df):
    """Histogram of UnitPrice on a log scale."""
    # Filter data to visual limits before binning to prevent outliers from compressing bins
    price_filtered = df["UnitPrice"][df["UnitPrice"] <= 20]
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(price_filtered, bins=80, color="darkorange", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Distribution of UnitPrice (log y-axis)", fontsize=13)
    ax.set_xlabel("Unit Price (£)")
    ax.set_ylabel("Frequency (log)")
    ax.set_xlim(0, 20)
    save(fig, "05_price_distribution.png")



def plot_top_products(df):
    """Horizontal bar chart: top 10 products by total revenue."""
    top = (
        df.groupby("Description")["Revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    sns.barplot(x=top.values, y=top.index, ax=ax, color="seagreen")
    ax.set_title("Top 10 Products by Revenue", fontsize=13)
    ax.set_xlabel("Total Revenue (£)")
    ax.set_ylabel("")
    save(fig, "06_top_products.png")


def plot_recency_distribution(features):
    """Histogram of customer Recency (days since last purchase)."""
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.hist(features["Recency"], bins=60, color="royalblue", edgecolor="white")
    ax.set_title("Customer Recency Distribution", fontsize=13)
    ax.set_xlabel("Days Since Last Purchase")
    ax.set_ylabel("Number of Customers")
    save(fig, "07_recency_distribution.png")


def plot_rfm_pairplot(features):
    """Pairplot of Recency, Frequency (log), and Monetary (log)."""
    # Log-transform Frequency and Monetary to reduce skew for visual clarity
    import numpy as np
    vis = features[["Recency", "Frequency", "Monetary"]].copy()
    vis["Frequency"]  = np.log1p(vis["Frequency"])
    vis["Monetary"]   = np.log1p(vis["Monetary"])
    vis = vis.rename(columns={
        "Frequency": "log(Frequency)",
        "Monetary":  "log(Monetary)"
    })

    g = sns.pairplot(vis, plot_kws={"alpha": 0.3, "s": 8}, diag_kind="hist")
    g.figure.suptitle("RFM Pairplot  (Frequency & Monetary log-transformed)",
                       y=1.02, fontsize=12)
    save(g.figure, "08_rfm_pairplot.png")


# -- Main -----------------------------------------------------------------------
if __name__ == "__main__":
    print("[INFO] Loading data …")
    df       = pd.read_csv(CLEAN_CSV, dtype={"CustomerID": str})
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    features = pd.read_csv(FEATURES_CSV, dtype={"CustomerID": str})

    print("[INFO] Generating figures …")
    plot_revenue_by_country(df)
    plot_monthly_revenue(df)
    plot_orders_by_weekday(df)
    plot_quantity_distribution(df)
    plot_price_distribution(df)
    plot_top_products(df)
    plot_recency_distribution(features)
    plot_rfm_pairplot(features)

    print(f"\n[INFO] All figures saved to {FIG_DIR}/")
