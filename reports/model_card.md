# Phase 6: Model Card & Ethical Reflection

This model card details the specifications, performance, constraints, and ethical considerations of the trained K-Means Customer Segmentation model.

---

## 1. Model Card

### Model Overview
*   **Model Name:** Customer RFM Clustered Profile Model
*   **Model Type:** K-Means Clustering Pipeline (Preprocessor + Estimator)
*   **Version:** 1.0.0
*   **Release Date:** June 17, 2026
*   **Trained By:** Hamza Amhidi & Mouaad Kaddouri
*   **Serialization File:** `models/best_clustering_pipeline.joblib`

### Intended Use
*   **Intended Domain:** E-commerce and retail transaction analytics.
*   **Primary Application:** Grouping customers into distinct value-based profiles to optimize marketing budget allocations.
*   **Supported Decisions:**
    *   *VIP Retention:* Identifying high-value customers for premium benefits.
    *   *Win-Back Campaigns:* Targeting inactive customers who previously had high engagement.
    *   *Upsell/Cross-sell:* Identifying frequent, low-spend buyers to invite to higher-value product tiers.
*   **Prohibited Applications:** Credit scoring, automated transaction blocking, fraud culpability, or individual price discrimination.

### Training Data & Preprocessing
*   **Primary Source:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail) containing transactional logs from a UK-based online retailer.
*   **Cohort Size:** 4,338 unique customers (extracted from 392,692 cleaned purchase rows).
*   **Feature Matrix Summary:**
    *   `Recency`: Continuous numeric (days since last order). Highly right-skewed.
    *   `Frequency`: Discrete integer (number of unique orders). Highly right-skewed.
    *   `Monetary`: Continuous float (total spent). Extremely right-skewed.
    *   `AvgOrderValue`: Continuous float (spent per order). Highly right-skewed.
    *   `UniqueProducts`: Discrete integer (distinct stock codes purchased).
    *   `ActiveDays`: Discrete integer (days between first and last purchase).
    *   `CancellationRate`: Continuous float (ratio of cancelled invoices, range [0, 1]).
*   **Preprocessing Pipeline:**
    1.  *Log-Transformation:* Apply `np.log1p` to `Recency`, `Frequency`, `Monetary`, `AvgOrderValue`, and `UniqueProducts` to normalize skewness.
    2.  *Robust Scaling:* Apply `RobustScaler` to scale features using median and IQR, preventing high-spend outliers from distorting distance weights.

### Performance Summary
The model selection was optimized using the Elbow Curve and Silhouette Coefficient across candidate values of $K = 2 \dots 8$. 

*   **Optimal Clusters ($K$):** 3 (Highest Silhouette Score)
*   **Performance Metrics:**
    *   *Silhouette Score:* **0.3093** (good separation and high cluster cohesion).
    *   *Davies-Bouldin Index:* **1.2021** (tightly bound clusters).
    *   *Calinski-Harabasz Score:* **1747.5** (variance ratio shows highly significant separation).
*   **Final Segment Distribution:**
    *   **Cluster 0 (At-Risk Customers):** 2,190 customers (Low recency, moderate spend).
    *   **Cluster 1 (VIP Customers):** 1,494 customers (High spend, high frequency, long history, low recency).
    *   **Cluster 2 (Loyal Customers):** 654 customers (Frequent buyers, moderate spend, active history, high cancellation rate).

### Model Constraints & Limitations
*   **Static Behavior:** The model represents a snapshot of customer transactions. Changes in purchase cycles or seasonal factors are not captured dynamically. The model must be retrained periodically (e.g. monthly).
*   **Outlier Retention:** Extremely high-spending outliers (e.g., corporate bulk buyers) are retained in the dataset. While `RobustScaler` prevents them from corrupting the centroids, they remain clustered alongside normal VIPs, which might skew average monetary values.

---

## 2. Ethical Reflection

### Data Privacy & Governance
*   **Lack of PII:** The model only trains on anonymous IDs and numeric transactional metrics. No customer names, email addresses, credit card numbers, or physical locations are ingested.
*   **Relational Storage Security:** The local SQLite database (`retail.db`) is stored in the `data/processed/` folder, which is locked under local system storage. It should not be uploaded to public clouds containing unencrypted customer keys.

### Fairness, Bias & Equity
*   **Bias in Marketing Exclusion:** Categorizing customers into segments like "Low-Value" or "At-Risk" can create feedback loops. If "Low-Value" customers are excluded from promotions, they will never transition to "Loyal" or "VIP" tiers, creating a self-fulfilling prophecy. Marketing teams should introduce randomized controls to verify promotions across all segments.
*   **Cancellation Rate Flagging:** The inclusion of `CancellationRate` as a feature flags customers who return items frequently. While crucial for logistics, these labels must not be used to automatically suspend accounts or deny customer service without manual oversight, as cancellations can occur due to shipping defects or sizing mistakes.
*   **Temporal Displacement Risk:** The training cohort spans Dec 2010 – Dec 2011. Customers labelled "At-Risk" based on historical recency may simply be from an earlier era of the business and no longer active for structural reasons (geography, product discontinuation). Segment labels should be communicated with this temporal caveat.

---

## 3. Deployment Checklist

Before using this model in a production marketing pipeline, the following conditions must be verified:

| Requirement | Status | Notes |
|---|---|---|
| Dataset refresh | Required monthly | Pipeline must ingest a new transactional export |
| Feature parity check | Automated | Assert all 7 features are present and non-null before inference |
| Scaler serialisation | Done | `RobustScaler` saved inside `best_clustering_pipeline.joblib` |
| Model version tag | Done | Version 1.0.0 logged in this card |
| Segment label audit | Manual | Marketing lead must review VIP/Loyal/At-Risk definitions quarterly |
| Outlier pre-screening | Recommended | Flag customers with Monetary > £50,000 for manual review before bulk email |
| Data residency | Required | `retail.db` must remain on-premise or in a GDPR-compliant cloud region |

---

## 4. Monitoring & Retraining Plan

| Event | Trigger | Action |
|---|---|---|
| Silhouette Score drops below 0.25 | Monthly re-evaluation run | Retrain model with new data; evaluate K again |
| Segment size imbalance exceeds 70/20/10 split | Monthly report | Re-evaluate K or re-label segments |
| New transaction features added (e.g., channel, geography) | Schema change event | Retrain full pipeline; update this model card |
| Business segment definition change | Stakeholder request | Update label heuristic in `train_cluster_models.py` and re-profile |

---

## 5. Regulatory Compliance Notes

*   **GDPR (EU 2016/679):** All training data uses pseudonymous `CustomerID` codes — no personal identifiers are processed. The pipeline complies with the data minimisation principle (Article 5(1)(c)).
*   **Right to Explanation:** Since K-Means is a transparent, distance-based model, each cluster assignment can be explained in plain language (e.g., "You were placed in the Loyal segment because your average order frequency is 31.6 orders and your last purchase was 48 days ago.").
*   **Automated Decision-Making:** This model must **not** be used as the sole basis for decisions that have significant legal or contractual effects on individuals (GDPR Article 22). All segment-based actions (promotions, account holds) require human review.

---

## 6. Revision History

| Version | Date | Author | Changes |
|---|---|---|---|
| 0.1.0 | 2026-06-14 | Hamza Amhidi & Mouaad Kaddouri | Initial K-Means baseline (no log transform) |
| 0.9.0 | 2026-06-14 | Hamza Amhidi & Mouaad Kaddouri | Added log1p + RobustScaler; fixed single-cluster collapse |
| 1.0.0 | 2026-06-17 | Hamza Amhidi & Mouaad Kaddouri | Final evaluation (K=3, Silhouette=0.3093); added deployment checklist, monitoring plan, and GDPR notes |

