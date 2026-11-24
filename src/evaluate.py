"""
Model evaluation module for Mall Customer Clustering.

Responsibilities:
1. Load raw customer data.
2. Select features from final_model configuration.
3. Encode categorical features when required.
4. Apply the configured scaler.
5. Train custom K-Means for different K values.
6. Calculate clustering evaluation metrics.
7. Save evaluation results.
8. Generate metric comparison visualizations.
9. Identify the best K according to each metric.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

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
from src.feature_processing import clean_records, prepare_feature_frame


# ============================================================
# 2. CENTRALIZED CONFIGURATION
# ============================================================

DATA_PATH = get_path("data", "raw_path")

RESULT_DIR = get_path("paths", "progress")

FIGURE_DIR = (
    get_path("paths", "figures")
    / "03_evaluation"
)

# Main model configuration
FINAL_CONFIG = get_config("final_model")

FEATURE_SPACE = FINAL_CONFIG["feature_space"]

FEATURE_COLUMNS = get_config(
    "experiments",
    "feature_spaces",
    FEATURE_SPACE
)

SCALER_NAME = FINAL_CONFIG["scaler"]

# Gender encoding
GENDER_MAPPING = get_config(
    "preprocessing",
    "encoding",
    "mapping",
    default={}
)

# Model parameters
MODEL_DEFAULTS = get_config("model")

K_VALUES = get_config(
    "experiments",
    "k_values"
)

RANDOM_STATE = FINAL_CONFIG.get(
    "random_state",
    MODEL_DEFAULTS.get("random_state", 42)
)

MAX_ITER = FINAL_CONFIG.get(
    "max_iterations",
    MODEL_DEFAULTS.get("max_iterations", 300)
)

TOLERANCE = FINAL_CONFIG.get(
    "tolerance",
    MODEL_DEFAULTS.get("tolerance", 0.0001)
)

INITIALIZATION = FINAL_CONFIG.get(
    "initialization",
    MODEL_DEFAULTS.get("initialization", "k-means++")
)

# Evaluation metrics
CONFIG_METRICS = get_config(
    "evaluation",
    "metrics",
    default=[]
)

SCALER_CLASSES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler
}


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


# ============================================================
# 4. VALIDATE CONFIGURATION
# ============================================================

if not K_VALUES:
    raise ValueError(
        "K values are missing in config.yaml."
    )

if any(
    not isinstance(k, int) or k < 2
    for k in K_VALUES
):
    raise ValueError(
        "All K values must be integers >= 2."
    )

if len(set(K_VALUES)) != len(K_VALUES):
    raise ValueError(
        "Duplicate K values detected."
    )

if FEATURE_SPACE not in (
    "with_gender",
    "without_gender"
):
    raise ValueError(
        f"Unsupported feature space: {FEATURE_SPACE}"
    )

if not FEATURE_COLUMNS:
    raise ValueError(
        "Feature columns are missing in config.yaml."
    )

if SCALER_NAME not in SCALER_CLASSES:
    raise ValueError(
        f"Unsupported scaler: {SCALER_NAME}"
    )

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
# 5. PREPARE DIRECTORIES
# ============================================================

RESULT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 6. LOAD AND PREPARE DATA
# ============================================================

def load_data():

    print("\n========== LOADING RAW DATA ==========")

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = clean_records(pd.read_csv(DATA_PATH))
    X = prepare_feature_frame(df, FEATURE_COLUMNS, GENDER_MAPPING)

    print("Dataset shape:", X.shape)

    print("Feature space:", FEATURE_SPACE)

    print("Features:", FEATURE_COLUMNS)

    print("Scaler:", SCALER_NAME)

    # Apply scaler
    scaler = SCALER_CLASSES[SCALER_NAME]()

    X_scaled = scaler.fit_transform(X)

    if not np.isfinite(X_scaled).all():

        raise ValueError(
            "Scaled data contains invalid values."
        )

    print("Scaling completed successfully.")

    return X_scaled


# ============================================================
# 7. CALCULATE METRICS
# ============================================================

def evaluate_model(X, labels):

    unique_labels = np.unique(labels)

    if (
        len(unique_labels) < 2
        or len(unique_labels) >= len(X)
    ):

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
# 8. RUN EVALUATION EXPERIMENTS
# ============================================================

def run_experiments(X):

    print("\n========== CLUSTERING EVALUATION ==========")

    results = []

    for k in K_VALUES:

        if k >= len(X):

            raise ValueError(
                f"K={k} must be smaller than sample count."
            )

        print(f"\nEvaluating K = {k}")

        model = KMeans(
            n_clusters=k,
            max_iter=MAX_ITER,
            tol=TOLERANCE,
            random_state=RANDOM_STATE,
            init=INITIALIZATION
        )

        labels = model.fit_predict(X)

        metrics = evaluate_model(
            X,
            labels
        )

        result = {

            "K": k,

            "Inertia": model.inertia_,

            **metrics,

            "Iterations": model.n_iter_,

            "Actual_Clusters": len(
                np.unique(labels)
            )

        }

        results.append(result)

        print(
            "Inertia:",
            round(model.inertia_, 4)
        )

        for metric_name in CONFIG_METRICS:

            column = METRIC_REGISTRY[
                metric_name
            ]["column"]

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
# 9. SAVE METRICS
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
# 10. VISUALIZE METRICS
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

    plt.title(
        f"{title}\n"
        f"{FEATURE_SPACE} - {SCALER_NAME}"
    )

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
# 11. ANALYZE BEST RESULTS
# ============================================================

def analyze_best_results(results):

    print("\n========== BEST METRIC RESULTS ==========")

    if results.empty:

        raise ValueError(
            "No evaluation results available."
        )

    for metric_name in CONFIG_METRICS:

        metric_config = METRIC_REGISTRY[
            metric_name
        ]

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

        print(
            "Score:",
            round(best[column], 4)
        )

    print(
        "\nNote: Compare all three metrics "
        "before selecting the final K."
    )


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n========================================")

    print("MALL CUSTOMER CLUSTERING EVALUATION")

    print("========================================")

    print("Feature space:", FEATURE_SPACE)

    print("Features:", FEATURE_COLUMNS)

    print("Scaler:", SCALER_NAME)

    print("K values:", list(K_VALUES))

    print("Random state:", RANDOM_STATE)

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

    print("\n========================================")

    print("EVALUATION COMPLETED SUCCESSFULLY!")

    print("========================================")


if __name__ == "__main__":
    main()
