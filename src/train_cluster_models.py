"""
train_cluster_models.py
=======================
Phase 4 - Clustering & Evaluation

Steps:
  1. Load customer features
  2. Select and scale clustering features (RobustScaler handles outliers well)
  3. Find the optimal K for K-Means using the Elbow + Silhouette method
  4. Train final K-Means model
  5. Evaluate with Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Score
  6. Assign human-readable segment labels to each cluster
  7. Save:
       - Labelled customer table  -> data/processed/customers_clustered.csv
       - Trained pipeline         -> models/best_clustering_pipeline.joblib
       - Elbow curve figure       -> reports/eda_figures/09_elbow_curve.png
       - Cluster profile figure   -> reports/eda_figures/10_cluster_profiles.png

Run: python src/train_cluster_models.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)

# -- Paths ----------------------------------------------------------------------
FEATURES_CSV   = "data/processed/customer_features.csv"
CLUSTERED_CSV  = "data/processed/customers_clustered.csv"
SUMMARY_CSV    = "data/processed/segment_summary.csv"
MODEL_PATH     = "models/best_clustering_pipeline.joblib"
FIG_DIR        = "reports/eda_figures"

# -- Features used for clustering ----------------------------------------------
# We include all 7 engineered features.
# RobustScaler uses the median and IQR, making it resistant to extreme outliers
# that are common in retail data (e.g. one customer with £200,000 in spend).
CLUSTER_FEATURES = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "UniqueProducts",
    "ActiveDays",
    "CancellationRate",
]

# -- Segment label mapping ------------------------------------------------------
# After training we profile each cluster by its feature means and assign a
# business-readable label.  The mapping below is applied by profile_clusters().
SEGMENT_MAP = {
    "vip":        "VIP Customers",
    "loyal":      "Loyal Customers",
    "at_risk":    "At-Risk Customers",
    "new":        "New Customers",
    "low_value":  "Low-Value Customers",
}


# -- Helper: Elbow + Silhouette search -----------------------------------------

def find_optimal_k(X_scaled: np.ndarray, k_range=range(2, 9)) -> int:
    """
    Iterate over candidate values of K and record:
      - Inertia          (for the Elbow plot)
      - Silhouette Score (higher is better, max = 1)

    Returns the K that maximises the Silhouette Score.
    """
    inertias, silhouettes = [], []

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))
        print(f"  k={k}  inertia={km.inertia_:,.0f}  silhouette={silhouettes[-1]:.4f}")

    # Plot Elbow curve
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(list(k_range), inertias, marker="o", color="steelblue")
    axes[0].set_title("Elbow Curve - Inertia vs K")
    axes[0].set_xlabel("Number of Clusters (K)")
    axes[0].set_ylabel("Inertia")

    axes[1].plot(list(k_range), silhouettes, marker="o", color="teal")
    axes[1].set_title("Silhouette Score vs K")
    axes[1].set_xlabel("Number of Clusters (K)")
    axes[1].set_ylabel("Silhouette Score")

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "09_elbow_curve.png"), dpi=150)
    plt.close(fig)
    print(f"  [Saved] {FIG_DIR}/09_elbow_curve.png")

    # Best K = highest silhouette
    best_k = list(k_range)[np.argmax(silhouettes)]
    print(f"\n[INFO] Best K by Silhouette Score: {best_k}")
    return best_k


# -- Helper: cluster profiling -------------------------------------------------

def profile_clusters(customers: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mean feature values per cluster and assign a business label.

    Labelling heuristic (applied in order of priority):
      VIP          - high Monetary AND high Frequency AND low Recency
      Loyal        - high Frequency AND low Recency (but not VIP monetary)
      At-Risk      - high Recency (long time since last purchase)
      New          - low ActiveDays AND low Frequency
      Low-Value    - everything else (low frequency, low monetary)
    """
    profile = customers.groupby("Cluster")[CLUSTER_FEATURES].mean()

    # Rank clusters on key dimensions (rank 0 = lowest)
    r_recency   = profile["Recency"].rank()         # lower Recency is better
    r_frequency = profile["Frequency"].rank(ascending=False)
    r_monetary  = profile["Monetary"].rank(ascending=False)
    r_active    = profile["ActiveDays"].rank(ascending=False)

    labels = {}
    for cluster in profile.index:
        if r_monetary[cluster] == 1 and r_frequency[cluster] <= 2:
            labels[cluster] = "VIP Customers"
        elif r_recency[cluster] >= len(profile) - 1:
            labels[cluster] = "At-Risk Customers"
        elif r_frequency[cluster] <= 2 and r_active[cluster] <= 2:
            labels[cluster] = "Loyal Customers"
        elif r_active[cluster] == len(profile):
            labels[cluster] = "New Customers"
        else:
            labels[cluster] = "Low-Value Customers"

    customers["Segment"] = customers["Cluster"].map(labels)
    return customers, profile


def plot_cluster_profiles(profile: pd.DataFrame):
    """
    Heatmap of normalised feature means per cluster.
    Normalising to 0-1 allows visual comparison across features with
    very different scales (e.g. Recency in days vs CancellationRate 0-1).
    """
    # Min-max normalise each column for display purposes only
    norm = (profile - profile.min()) / (profile.max() - profile.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(
        norm.T,
        annot=profile.T.round(1),   # show actual values in each cell
        fmt="g",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Cluster Profiles - Normalised Feature Means", fontsize=13)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    path = os.path.join(FIG_DIR, "10_cluster_profiles.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  [Saved] {path}")


# -- Main -----------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)
    os.makedirs(FIG_DIR,  exist_ok=True)

    # 1. Load features
    customers = pd.read_csv(FEATURES_CSV, dtype={"CustomerID": str})
    X = customers[CLUSTER_FEATURES].copy()

    # 2. Scale  (RobustScaler: uses median & IQR, ignores extreme outliers)
    scaler   = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Find best K
    print("\n-- K search -----------------------------------------------")
    best_k = find_optimal_k(X_scaled)

    # 4. Train final model
    print(f"\n-- Training K-Means with K={best_k} -----------------------")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    customers["Cluster"] = kmeans.fit_predict(X_scaled)

    # 5. Evaluation metrics
    sil = silhouette_score(X_scaled, customers["Cluster"])
    dbi = davies_bouldin_score(X_scaled, customers["Cluster"])
    chi = calinski_harabasz_score(X_scaled, customers["Cluster"])

    print(f"\n-- Evaluation metrics -------------------------------------")
    print(f"  Silhouette Score        : {sil:.4f}  (higher -> better separation)")
    print(f"  Davies-Bouldin Index    : {dbi:.4f}  (lower  -> better separation)")
    print(f"  Calinski-Harabasz Score : {chi:.1f}  (higher -> better separation)")

    # 6. Profile & label clusters
    customers, profile = profile_clusters(customers)

    print("\n-- Cluster sizes ------------------------------------------")
    print(customers.groupby(["Cluster", "Segment"]).size()
          .reset_index(name="Customers").to_string(index=False))

    # 7. Save outputs
    customers.to_csv(CLUSTERED_CSV, index=False)
    print(f"\n[INFO] Clustered customers saved to {CLUSTERED_CSV}")

    # Bundle scaler + model into a single Pipeline for reproducibility
    pipeline = Pipeline([
        ("scaler",  scaler),
        ("kmeans",  kmeans),
    ])
    joblib.dump(pipeline, MODEL_PATH)
    print(f"[INFO] Model pipeline saved to {MODEL_PATH}")

    # 8. Visualise cluster profiles
    plot_cluster_profiles(profile)
