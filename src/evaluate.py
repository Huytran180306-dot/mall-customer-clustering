"""
Model evaluation module for Mall Customer Clustering.

Responsibilities:
1. Load standardized customer data.
2. Train custom K-Means for different K values.
3. Calculate clustering evaluation metrics.
4. Save evaluation results.
5. Generate metric comparison visualizations.
6. Identify the best K according to each metric.
"""

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


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_config, get_path
from src.models.kmeans import KMeans


# ============================================================
# 2. LOAD CENTRALIZED CONFIGURATION
# ============================================================

DATA_PATH = (
    get_path("data", "processed_dir")
    / "Mall_Customers_StandardScaler.csv"
)

RESULT_DIR = get_path("paths", "progress")

FIGURE_DIR = (
    get_path("paths", "figures")
    / "03_evaluation"
)

K_VALUES = get_config("experiments", "k_values")

RANDOM_STATE = get_config(
    "model", "random_state", default=42
)

MAX_ITER = get_config(
    "model", "max_iterations", default=300
)

TOLERANCE = get_config(
    "model", "tolerance", default=0.0001
)

INITIALIZATION = get_config(
    "model", "initialization", default="k-means++"
)

CONFIG_METRICS = get_config(
    "evaluation", "metrics"
)


# ============================================================
# 3. METRIC REGISTRY
# ============================================================

METRIC_REGISTRY = {

    "silhouette_score": {
        "column": "Silhouette_Score",
        "function": silhouette_score,
        "direction": "max"
    },

    "davies_bouldin_score": {
        "column": "Davies_Bouldin_Index",
        "function": davies_bouldin_score,
        "direction": "min"
    },

    "calinski_harabasz_score": {
        "column": "Calinski_Harabasz_Score",
        "function": calinski_harabasz_score,
        "direction": "max"
    }

}

REQUIRED_METRICS = {
    "silhouette_score",
    "davies_bouldin_score",
    "calinski_harabasz_score"
}

if not CONFIG_METRICS:
    raise ValueError(
        "Evaluation metrics are missing in config.yaml."
    )

if not REQUIRED_METRICS.issubset(set(CONFIG_METRICS)):
    raise ValueError(
        "Silhouette, Davies-Bouldin and "
        "Calinski-Harabasz metrics are required."
    )

for metric in CONFIG_METRICS:
    if metric not in METRIC_REGISTRY:
        raise ValueError(
            f"Unsupported evaluation metric: {metric}"
        )


# ============================================================
# 4. PREPARE DIRECTORIES
# ============================================================

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5. LOAD DATA
# ============================================================

def load_data():

    print("\n========== LOADING DATA ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    if df.empty:
        raise ValueError(
            "Evaluation dataset is empty."
        )

    X = df.to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            "Dataset contains missing or infinite values."
        )

    print("Dataset shape:", X.shape)

    print("Features:", df.columns.tolist())

    return X


# ============================================================
# 6. CALCULATE METRICS
# ============================================================

def evaluate_model(X, labels):

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2 or len(unique_labels) >= len(X):
        raise ValueError(
            "Invalid number of clusters for evaluation."
        )

    metrics = {}

    for metric_name in CONFIG_METRICS:

        metric_config = METRIC_REGISTRY[metric_name]

        metric_function = metric_config["function"]

        column_name = metric_config["column"]

        metrics[column_name] = float(
            metric_function(X, labels)
        )

    return metrics


# ============================================================
# 7. RUN EVALUATION EXPERIMENTS
# ============================================================

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
            init=INITIALIZATION
        )

        labels = model.fit_predict(X)

        metrics = evaluate_model(X, labels)

        result = {

            "K": k,

            "Inertia": model.inertia_,

            **metrics,

            "Iterations": model.n_iter_,

            "Actual_Clusters": len(np.unique(labels))

        }

        results.append(result)

        print("Inertia:", round(model.inertia_, 4))

        for metric_name in CONFIG_METRICS:

            column = METRIC_REGISTRY[metric_name]["column"]

            print(
                f"{column}:",
                round(metrics[column], 4)
            )

    results_df = pd.DataFrame(results)

    if len(results_df) != len(K_VALUES):
        raise RuntimeError(
            "Evaluation experiment count mismatch."
        )

    return results_df


# ============================================================
# 8. SAVE METRICS
# ============================================================

def save_results(results):

    output_path = (
        RESULT_DIR / "clustering_metrics.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nMetrics saved:", output_path)


# ============================================================
# 9. VISUALIZE METRICS
# ============================================================

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


# ============================================================
# 10. ANALYZE BEST RESULTS
# ============================================================

def analyze_best_results(results):

    print("\n========== BEST METRIC RESULTS ==========")

    if results.empty:
        raise ValueError(
            "No evaluation results available."
        )

    for metric_name in CONFIG_METRICS:

        metric_config = METRIC_REGISTRY[metric_name]

        column = metric_config["column"]

        direction = metric_config["direction"]

        if direction == "max":

            best = results.loc[
                results[column].idxmax()
            ]

        else:

            best = results.loc[
                results[column].idxmin()
            ]

        print(f"\nMetric: {column}")

        print("Best K:", int(best["K"]))

        print("Score:", round(best[column], 4))

    print(
        "\nNote: Compare all three metrics "
        "before selecting the final K."
    )


# ============================================================
# 11. MAIN
# ============================================================

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