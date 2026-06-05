# Customer Segmentation – RFM + K-Means Clustering

**Team:** Hamza Amhidi & Mouaad Kaddouri  
**Module:** AI & Data Science Basics – EHTP, S4  
**Dataset:** [UCI Online Retail](https://archive.ics.uci.edu/dataset/352/online+retail)

---

## Problem

Group ~4,300 customers of a UK-based online retailer into interpretable behavioral
segments (VIP, Loyal, At-Risk, New, Low-Value) using RFM features and K-Means clustering.

## Repository structure

```
customer-segmentation-rfm/
├── data/
│   ├── raw/               Online Retail Excel + raw CSV
│   └── processed/         Cleaned CSV, feature table, clustered CSV, SQLite DB
├── src/
│   ├── load_describe_data.py   Phase 1 – download & audit
│   ├── clean_data.py           Phase 2 – wrangling
│   ├── features.py             Phase 3 – RFM feature engineering
│   ├── eda.py                  Phase 3 – 8 EDA figures
│   ├── store_sql.py            Phase 3 – SQLite storage + 5 SQL queries
│   └── train_cluster_models.py Phase 4 – K-Means + evaluation
├── models/
│   └── best_clustering_pipeline.joblib
├── reports/
│   └── eda_figures/            10 PNG figures
├── run_pipeline.py             Run the full pipeline end-to-end
└── requirements.txt
```

## Reproduce

```bash
pip install -r requirements.txt
python run_pipeline.py
```

The script downloads the dataset automatically from UCI on the first run.

To run the pipeline with your own transaction file:

```bash
python run_pipeline.py --input path/to/your_transactions.csv
```

CSV and Excel inputs are supported. Your file must contain these columns:

| Column | Meaning |
|---|---|
| InvoiceNo | Order/invoice identifier |
| StockCode | Product identifier |
| Description | Product name or description |
| Quantity | Quantity purchased |
| InvoiceDate | Purchase date/time |
| UnitPrice | Product unit price |
| CustomerID | Customer identifier |
| Country | Customer/order country |

## Features engineered

| Feature | Description |
|---|---|
| Recency | Days since last purchase |
| Frequency | Unique invoice count |
| Monetary | Total spend (£) |
| AvgOrderValue | Monetary / Frequency |
| UniqueProducts | Distinct products purchased |
| ActiveDays | Days between first and last purchase |
| CancellationRate | Fraction of cancelled invoices |

## Citation

Chen, D. (2015). Online Retail [Dataset]. UCI Machine Learning Repository.  
DOI: [10.24432/C5BW33](https://doi.org/10.24432/C5BW33)
