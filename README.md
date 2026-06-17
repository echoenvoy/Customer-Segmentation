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
│   ├── eda.py                  Phase 3b – 10 EDA figures
│   ├── store_sql.py            Phase 3c – SQLite storage + 5 SQL queries
│   └── train_cluster_models.py Phase 5 – K-Means + evaluation
├── models/
│   └── best_clustering_pipeline.joblib   Serialised log1p + RobustScaler + KMeans
├── notebooks/
│   └── customer_segmentation.ipynb       End-to-end evaluation notebook (Phases 0–6)
├── reports/
│   ├── algorithm_selection.md            Phase 4 written report
│   ├── model_card.md                     Phase 6 model card
│   ├── phase5_clustering_evaluation.md   Phase 5 clustering results
│   └── eda_figures/                      10 high-resolution PNG charts
├── run_pipeline.py             MASTER RUNNER – executes full pipeline
├── requirements.txt            Package dependencies
└── .gitignore                  Excludes caches and virtual environments
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

Using log-transformed RFM features, the K-Means algorithm (optimised at K=3) produces three well-separated customer segments:

| Segment | Count | % | Avg Recency | Avg Monetary | Strategy |
|---|---|---|---|---|---|
| **VIP** | 412 | 9.5% | 22 days | £8,521 | Exclusive loyalty rewards, early access |
| **Loyal** | 1,847 | 42.6% | 48 days | £1,238 | Upsell campaigns, volume discounts |
| **At-Risk** | 2,079 | 47.9% | 187 days | £298 | Win-back emails, personalised vouchers |

### Best Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| **Silhouette Score** | **0.3093** | Good cluster cohesion and separation |
| **Davies-Bouldin Index** | **1.2021** | Well-separated clusters (< 1.5 threshold) |
| **Calinski-Harabasz Score** | **2,341.8** | High between/within cluster variance ratio |
| **Optimal K** | **3** | Verified by Elbow curve + 3 independent metrics |

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
Open and run all cells in `notebooks/customer_segmentation.ipynb`.

> **Google Colab users:** Click the Colab badge at the top. Run **Phase 0 first** — it clones the repo and installs dependencies automatically.

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
