# Phase 5 – Machine Learning Clustering & Segment Profiling

**Project:** Online Retail Customer Segmentation  
**Authors:** Hamza Amhidi & Mouaad Kaddouri  
**Date:** June 16, 2026

---

## 1. Preprocessing Pipeline

Before clustering, all 7 customer-level features undergo a two-step transformation:

### 1.1 Log1p Transformation (Skew Reduction)

Five features exhibit strong right-skew (long-tail distributions dominated by top spenders). Applying `numpy.log1p` compresses the tails while preserving zero values:

| Feature | Raw Skewness | Post-log1p Skewness |
|---|---|---|
| Recency | 1.83 | 0.61 |
| Frequency | 7.42 | 0.91 |
| Monetary | 9.17 | 0.74 |
| AvgOrderValue | 8.05 | 0.68 |
| UniqueProducts | 5.29 | 0.87 |

`ActiveDays` and `CancellationRate` were left untransformed (already near-normal).

### 1.2 RobustScaler Normalization

After log-transformation, `sklearn.preprocessing.RobustScaler` scales each feature using its **median and interquartile range (IQR)** instead of mean/std. This ensures that any remaining outliers do not distort the cluster centroids.

---

## 2. Optimal K Selection

K-Means was evaluated for K = 2 through 8 using three complementary metrics:

| K | Silhouette Score ↑ | Davies-Bouldin Index ↓ | Calinski-Harabasz ↑ |
|---|---|---|---|
| 2 | 0.2741 | 1.6120 | 1847.3 |
| **3** | **0.3093** | **1.2021** | **2341.8** |
| 4 | 0.2815 | 1.4302 | 2198.5 |
| 5 | 0.2590 | 1.5887 | 2051.2 |
| 6 | 0.2412 | 1.7001 | 1923.4 |
| 7 | 0.2198 | 1.8230 | 1803.7 |
| 8 | 0.2054 | 1.9511 | 1701.9 |

> **Chosen K = 3** — highest Silhouette Score (0.3093), lowest Davies-Bouldin Index (1.2021), and highest Calinski-Harabasz Score (2341.8).

The elbow curve (WCSS vs K) confirms a pronounced elbow at K = 3, after which additional clusters yield diminishing returns.

---

## 3. Cluster Profiles

After fitting the final K-Means model (`random_state=42`, `n_init=10`), each cluster was profiled by computing the **mean raw feature values** per segment. Clusters were then uniquely assigned business labels using a priority-based heuristic:

- **VIP** → cluster with the **highest Monetary** value (highest spenders)
- **Loyal** → cluster with the **lowest Recency** among the remaining two (most recently active)
- **At-Risk** → the remaining cluster (high recency = disengaged customers)

### Cluster Centroids (Raw Feature Space)

| Segment | Customers | Recency (days) | Frequency (orders) | Monetary (£) | AvgOrderValue (£) | UniqueProducts | ActiveDays | CancellationRate |
|---|---|---|---|---|---|---|---|---|
| **VIP** | 412 | 22.4 | 87.3 | 8 521.4 | 97.6 | 312.1 | 295.2 | 0.082 |
| **Loyal** | 1 847 | 48.1 | 31.6 | 1 237.8 | 39.2 | 98.4 | 201.7 | 0.061 |
| **At-Risk** | 2 079 | 187.3 | 8.1 | 298.3 | 36.9 | 35.7 | 64.3 | 0.043 |

### Interpretation

| Segment | Behavioural Profile | Recommended Action |
|---|---|---|
| **VIP** (9.5%) | High-frequency, high-spend repeat buyers who have purchased very recently. Buy a wide product variety. | Exclusive loyalty rewards, early access to new collections, personal account managers. |
| **Loyal** (42.6%) | Regular buyers with moderate spend and recent activity. Core revenue backbone. | Upselling campaigns, volume discounts, mid-tier loyalty programme enrolment. |
| **At-Risk** (47.9%) | Infrequent buyers with very high recency (>6 months since last order). Lowest spend. | Win-back email sequences, personalised discount vouchers, re-engagement surveys. |

---

## 4. Evaluation Summary

| Metric | Value | Interpretation |
|---|---|---|
| Silhouette Score | **0.3093** | Moderate–good cohesion; clusters are meaningfully separated. |
| Davies-Bouldin Index | **1.2021** | Below 1.5 threshold; clusters are distinct with limited overlap. |
| Calinski-Harabasz Score | **2341.8** | Strong between-cluster variance; tight and well-separated partitions. |
| Inertia (WCSS) | **8 412.7** | Stable after K = 3; further clusters do not reduce variance significantly. |

---

## 5. Artefacts Produced

| File | Description |
|---|---|
| `data/processed/customers_clustered.csv` | Full customer dataset with `Cluster` (0–2) and `Segment` (VIP/Loyal/At-Risk) columns. |
| `models/best_clustering_pipeline.joblib` | Serialised pipeline: `(log1p_transformer, RobustScaler, KMeans(K=3))`. |
| `reports/eda_figures/09_elbow_curve.png` | WCSS elbow chart for K = 2–8. |
| `reports/eda_figures/10_cluster_profiles.png` | Heatmap of normalised cluster feature means. |
