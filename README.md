# Customer Segmentation – RFM + K-Means Clustering

**Team:** Hamza Amhidi & Mouaad Kaddouri  
**Dataset:** [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

---

##  Quick Web Evaluation

For easy grading and review, you can evaluate the project directly in the web browser :

*   [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/echoenvoy/Customer-Segmentation/blob/main/notebooks/customer_segmentation.ipynb) **Run in Google Colab:** Open and execute the notebook in Google's cloud-hosted environment.
*   [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/echoenvoy/Customer-Segmentation/main?labpath=notebooks%2Fcustomer_segmentation.ipynb) **Interactive Environment (Binder):** Launch a live JupyterLab environment to execute and experiment with all notebook cells.


---

## Overview

This repository contains an end-to-end customer analytics and segmentation pipeline. It processes transactional sales logs, cleans and filters them, extracts customer behavioral features, saves them into an SQLite database for queries, trains an unsupervised machine learning clustering model (K-Means), and profiles the resulting customer segments (VIP, Loyal, At-Risk) for marketing actions.

## Key Program Updates

We have resolved several bugs and optimized the machine learning pipeline:
1. **Mathematical Bug Fix in Cancellation Rate:** Previously, the cancellation rate divided row-level transaction counts by unique invoice counts, leading to impossible rates (e.g. 750%). It is now correctly computed at the unique invoice level (cancelled unique invoices / total unique invoices).
2. **Clustering Outlier Mitigation:** To prevent high-spending outliers from collapsing all customers into a single cluster (the "2-customer VIP" problem), we apply a `log1p` transformation to right-skewed variables (`Recency`, `Frequency`, `Monetary`, `AvgOrderValue`, and `UniqueProducts`) before scaling.
3. **Robust Labeling Heuristic:** We implemented a deterministic, unique labeling hierarchy to ensure that each cluster is mapped to a unique segment name without duplicate assignments.
4. **Performance Boost:** The runner script now checks if the raw CSV already exists on subsequent runs, skipping the slow Excel parsing step and reducing launch times by ~25 seconds.

---

## Repository Structure

```
customer-segmentation-rfm/
├── data/
│   ├── raw/                    Online Retail raw CSV + Excel
│   └── processed/              Cleaned CSV, feature table, clustered CSV, SQLite DB
├── src/
│   ├── load_describe_data.py   Phase 1 – download & audit
│   ├── clean_data.py           Phase 2 – wrangling & cleaning
│   ├── features.py             Phase 3 – RFM feature engineering
│   ├── eda.py                  Phase 3b – 8 EDA figures
│   ├── store_sql.py            Phase 3c – SQLite storage + 5 SQL queries
│   └── train_cluster_models.py Phase 4 – K-Means + evaluation
├── models/
│   └── best_clustering_pipeline.joblib   - Serialized preprocessor + K-Means model
├── reports/
│   └── eda_figures/            10 PNG plots and charts
├── customer_segmentation.ipynb  - STEP-BY-STEP EVALUATION NOTEBOOK
├── run_pipeline.py             - MASTER RUNNER (Runs full pipeline)
├── requirements.txt            - Package dependencies
└── .gitignore                  - Excludes data/models/cache files
```

---

## Features Engineered

| Feature | Description |
|---|---|
| **Recency** | Days since last purchase |
| **Frequency** | Unique invoice (order) count |
| **Monetary** | Total spend (£) |
| **AvgOrderValue** | Monetary / Frequency |
| **UniqueProducts** | Distinct products purchased |
| **ActiveDays** | Days between first and last purchase |
| **CancellationRate** | Fraction of cancelled invoices |

---

## Customer Segments Profile

Using log-transformed RFM features, the K-Means algorithm (optimized at K=3) maps customers to balanced, distinct behavioral profiles:

*   **VIP Customers (Cluster 1):** Highly active, high-spending, and very recent purchasers.
    *   *Strategy:* Exclusive rewards, early product access, and high-touch support.
*   **Loyal Customers (Cluster 2):** Frequent buyers but with moderate spend.
    *   *Strategy:* Upsell programs, loyalty points, and bundle offers.
*   **At-Risk Customers (Cluster 0):** Dormant purchasers who haven't bought in a long time.
    *   *Strategy:* Win-back campaigns, personalized discounts, and check-ins.

---

## How to Evaluate and Run the Project

### 1. Installation
Install the required packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Interactive Notebook (Recommended)
Launch JupyterLab to interactively evaluate the project step-by-step with inline plots and descriptions:
```bash
jupyter lab
```
Open and run all cells in [customer_segmentation.ipynb](file:///c:/Users/Mouaad/Documents/Ai-project/Customer-Segmentation/customer_segmentation.ipynb).

### 3. Run the Automated Pipeline
To execute the pipeline from acquisition to model training and print query outputs:
```bash
python run_pipeline.py
```

To run it with your own transaction CSV/Excel file:
```bash
python run_pipeline.py --input path/to/your_transactions.csv
```
*(Your input file must contain columns: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, `Country`)*
