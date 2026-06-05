# Project Summary & Feature Catalog

This document provides a detailed definition of the **Customer Segmentation Pipeline** and a comprehensive guide to all features and scripts built into the program.

---

## Project Definition
This program is an end-to-end customer analytics and segmentation pipeline. It processes transactional sales logs, cleans and filters them, extracts customer behavioral features, saves them into an SQLite database for queries, trains an unsupervised machine learning clustering model (K-Means), and profiles the resulting customer segments (VIP, Loyal, At-Risk, New, Low-Value) for marketing actions.

---

## Complete Features of the Program

### 1. Flexible Data Input & Master Runner
* **Single Entry Point:** `run_pipeline.py` executes all stages of the pipeline automatically from download to model evaluation.
* **Custom Dataset Parsing:** Supports `--input` CLI parameters to let users run the pipeline on custom `.csv`, `.xlsx`, or `.xls` transaction sheets:
  ```bash
  python run_pipeline.py --input path/to/transactions.csv
  ```
* **Required Input Columns:** The script validates that the input dataset has the following headers: `InvoiceNo`, `StockCode`, `Description`, `Quantity`, `InvoiceDate`, `UnitPrice`, `CustomerID`, and `Country`.

### 2. Demo Dataset Generation
* **Synthetic Data Creator:** `generate_demo_data.py` generates a synthetic dataset containing 500,000 transaction rows and 4,000 unique customer IDs.
* **Pipeline Testing:** Simulates negative quantities (to test cancellations), missing customer IDs, and unit prices to test the pipeline end-to-end without downloading the actual dataset.

### 3. Automatic Data Acquisition & Quality Auditing
* **Auto-Acquisition:** `src/load_describe_data.py` automatically downloads the zipped Online Retail Excel file from the UCI Machine Learning Repository, extracts it, and converts it to a standard CSV.
* **Automated Audit Report:** Generates a detailed stdout quality audit report detailing:
  * Rows and columns dimensions.
  * Count and percentage of missing values.
  * Count of exact duplicate rows.
  * Count of cancelled transactions.
  * Statistics on negative or zero prices/quantities.
  * Basic descriptive statistics of the numerical attributes.

### 4. Robust Data Cleaning
* **Cleaning Pipeline:** `src/clean_data.py` cleans the transaction dataset, reporting rows kept/removed at each step:
  * Drops exact duplicate rows.
  * Drops transactions without a `CustomerID`.
  * Removes invoice cancellations (prefixed with 'C') so they don't corrupt monetary and frequency spend metrics (they are instead calculated as a separate cancellation rate feature).
  * Excludes non-positive quantities ($Quantity \le 0$) and unit prices ($UnitPrice \le 0$).
  * Appends a calculated `Revenue` column ($Quantity \times UnitPrice$) to each transaction.

### 5. Multi-Dimensional Feature Engineering
* **Customer Profiler:** `src/features.py` aggregates line-item transactions into a per-customer feature matrix:
  * **Recency:** Days between the customer's last transaction and a dataset reference date (max date + 1 day).
  * **Frequency:** Total number of unique orders placed by the customer.
  * **Monetary:** Sum of revenue spent across all orders.
  * **AvgOrderValue:** Average amount spent per order.
  * **UniqueProducts:** Count of distinct product codes purchased.
  * **ActiveDays:** Number of days between the customer's first and last order.
  * **CancellationRate:** Calculated by dividing cancelled invoices by total invoices from the raw dataset, measuring return risk.

### 6. Relational SQL Database Storage & Queries
* **Database Manager:** `src/store_sql.py` establishes a local SQLite database (`data/processed/retail.db`) with tables:
  * `transactions`: Cleaned line-item purchases.
  * `customers`: Engineered feature profiles.
* **Analytical Queries:** Executes and outputs results of 5 complex queries:
  * **Q1:** Top 10 customer spending champions.
  * **Q2:** Total revenue, orders, and unique customer counts by country.
  * **Q3:** Monthly revenue and order trends.
  * **Q4:** Overall average RFM metrics.
  * **Q5:** High-spend customers with a return rate greater than 20% ("Risky VIPs").

### 7. Interactive Exploratory Visualizations
* **EDA Suite:** `src/eda.py` produces 8 high-resolution charts saved in `reports/eda_figures/`:
  1. *Top Countries by Revenue* (excluding United Kingdom to prevent chart distortion).
  2. *Monthly Revenue Trend* line graph.
  3. *Day of Week Order Distribution* bar chart.
  4. *Quantity Distribution histogram* (log-scaled).
  5. *UnitPrice Distribution histogram* (log-scaled).
  6. *Top 10 Products by Revenue* horizontal bar chart.
  7. *Customer Recency Distribution histogram*.
  8. *RFM Pairplot* (using log-transformed variables to account for right-skewed data).

### 8. Machine Learning Clustering & Segment Profiling
* **Outlier-Resistant Preprocessing:** `src/train_cluster_models.py` scales the data using a `RobustScaler` (median-based), which prevents high-spending outliers from skewing cluster boundaries.
* **Optimal K Search:** Loops through potential numbers of clusters ($K=2$ to $8$) to calculate inertia (Elbow Curve) and Silhouette Scores, saving the visualization to `reports/eda_figures/09_elbow_curve.png`.
* **Model Evaluation:** Computes and prints performance metrics:
  * *Silhouette Score* (higher indicates better cluster cohesion and separation).
  * *Davies-Bouldin Index* (lower indicates tighter clusters).
  * *Calinski-Harabasz Score* (higher indicates better-defined clusters).
* **Customer Segment Mapping:** Automatically maps numerical clusters to business-focused categories:
  * **VIP Customers:** Low Recency, high Frequency, and high Monetary spend.
  * **Loyal Customers:** Low Recency, high Frequency, but moderate Monetary spend.
  * **At-Risk Customers:** High Recency (haven't bought in a long time).
  * **New Customers:** Low ActiveDays and low Frequency (recently registered).
  * **Low-Value Customers:** Low Frequency and low Monetary spend.
* **Heatmap Profiling:** Saves a normalized feature heatmap (`reports/eda_figures/10_cluster_profiles.png`) showing actual feature means for each segment.
* **Serialized Model Pipeline:** Saves the complete preprocessor and trained K-Means model to `models/best_clustering_pipeline.joblib` for reproducibility.
