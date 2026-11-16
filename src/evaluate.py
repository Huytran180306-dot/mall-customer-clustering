import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# Add project root to Python path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.kmeans import KMeans


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = (
    ROOT_DIR /
    "data/processed/Mall_Customers_StandardScaler.csv"
)

RESULT_DIR = ROOT_DIR / "progress_results"

FIGURE_DIR = (
    ROOT_DIR /
    "reports/figures/customer_segmentation/03_evaluation"
)

K_VALUES = range(2, 11)

RANDOM_STATE = 42
MAX_ITER = 300
TOLERANCE = 0.0001

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    print("\n========== LOADING DATA ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    X = df.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            "Dataset contains missing or infinite values."
        )

    print("Dataset shape:", X.shape)

    return X


# ==========================================
# 3. CALCULATE METRICS
# ==========================================

def evaluate_model(X, labels):

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        raise ValueError(
            "Invalid number of clusters for evaluation."
        )

    silhouette = silhouette_score(X, labels)

    dbi = davies_bouldin_score(X, labels)

    ch = calinski_harabasz_score(X, labels)

    return silhouette, dbi, ch


# ==========================================
# 4. RUN EVALUATION EXPERIMENTS
# ==========================================

def run_experiments(X):

    print("\n========== CLUSTERING EVALUATION ==========")

    results = []

    for k in K_VALUES:

        print(f"\nEvaluating K = {k}")

        model = KMeans(
            n_clusters=k,
            max_iter=MAX_ITER,
            tol=TOLERANCE,
            random_state=RANDOM_STATE,
            init="k-means++"
        )

        model.fit(X)

        labels = model.labels_

        silhouette, dbi, ch = evaluate_model(
            X,
            labels
        )

        results.append({

            "K": k,

            "Inertia": model.inertia_,

            "Silhouette_Score": silhouette,

            "Davies_Bouldin_Index": dbi,

            "Calinski_Harabasz_Score": ch,

            "Iterations": model.n_iter_,

            "Actual_Clusters": len(np.unique(labels))

        })

        print("Inertia:", round(model.inertia_, 4))

        print("Silhouette:", round(silhouette, 4))

        print("Davies-Bouldin:", round(dbi, 4))

        print("Calinski-Harabasz:", round(ch, 4))

    return pd.DataFrame(results)


# ==========================================
# 5. SAVE METRICS
# ==========================================

def save_results(results):

    output_path = (
        RESULT_DIR / "clustering_metrics.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nMetrics saved:", output_path)


# ==========================================
# 6. VISUALIZE METRICS
# ==========================================

def plot_metric(
    results,
    metric,
    title,
    ylabel,
    filename
):

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["K"],
        results[metric],
        marker="o"
    )

    plt.title(title)

    plt.xlabel("Number of Clusters (K)")

    plt.ylabel(ylabel)

    plt.xticks(list(K_VALUES))

    plt.grid(True)

    plt.tight_layout()

    output_path = FIGURE_DIR / filename

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Figure saved:", output_path)


# ==========================================
# 7. ANALYZE BEST RESULTS
# ==========================================

def analyze_best_results(results):

    print("\n========== BEST METRIC RESULTS ==========")

    best_silhouette = results.loc[
        results["Silhouette_Score"].idxmax()
    ]

    best_dbi = results.loc[
        results["Davies_Bouldin_Index"].idxmin()
    ]

    best_ch = results.loc[
        results["Calinski_Harabasz_Score"].idxmax()
    ]

    print(
        "Best Silhouette K:",
        int(best_silhouette["K"])
    )

    print(
        "Best Davies-Bouldin K:",
        int(best_dbi["K"])
    )

    print(
        "Best Calinski-Harabasz K:",
        int(best_ch["K"])
    )

    print(
        "\nNote: Compare all three metrics "
        "before selecting the final K."
    )


# ==========================================
# 8. MAIN
# ==========================================

def main():

    print("\n================================")

    print("MALL CUSTOMER CLUSTERING EVALUATION")

    print("================================")

    X = load_data()

    results = run_experiments(X)

    print("\n========== RESULTS SUMMARY ==========")

    print(
        results.round(4).to_string(index=False)
    )

    save_results(results)

    plot_metric(
        results,
        "Silhouette_Score",
        "Silhouette Score Comparison",
        "Silhouette Score",
        "silhouette_score.png"
    )

    plot_metric(
        results,
        "Davies_Bouldin_Index",
        "Davies-Bouldin Index Comparison",
        "Davies-Bouldin Index",
        "davies_bouldin_index.png"
    )

    plot_metric(
        results,
        "Calinski_Harabasz_Score",
        "Calinski-Harabasz Score Comparison",
        "Calinski-Harabasz Score",
        "calinski_harabasz_score.png"
    )

    analyze_best_results(results)

    print("\n================================")

    print("EVALUATION COMPLETED SUCCESSFULLY!")

    print("================================")


if __name__ == "__main__":
    main()