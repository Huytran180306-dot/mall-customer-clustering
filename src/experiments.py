"""
Experiments module for Mall Customer Clustering.

Responsibilities:
1. Load and validate raw customer data.
2. Run 36 K-Means experiments.
3. Compare feature spaces and scaling methods.
4. Evaluate clustering using internal metrics.
5. Save experimental results and visualizations.
6. Summarize the best configurations.
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
# 2. LOAD CENTRALIZED CONFIGURATION
# ============================================================

DATA_PATH = get_path("data", "raw_path")

RESULT_DIR = get_path("paths", "progress")

FIGURE_DIR = (
    get_path("paths", "figures")
    / "05_extended"
    / "experiments"
)

K_VALUES = get_config("experiments", "k_values")

RANDOM_STATE = get_config(
    "model", "random_state",
    default=get_config("project", "random_state", default=42)
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

GENDER_MAPPING = get_config(
    "preprocessing", "encoding", "mapping"
)

CONFIG_FEATURE_SPACES = get_config(
    "experiments", "feature_spaces"
)

CONFIG_SCALERS = get_config(
    "experiments", "scalers"
)

CONFIG_METRICS = get_config(
    "evaluation", "metrics"
)


# ============================================================
# 3. SCALER AND METRIC REGISTRY
# ============================================================

SCALER_CLASSES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler
}

METRIC_REGISTRY = {
    "silhouette_score": (
        "Silhouette_Score",
        silhouette_score,
        "max"
    ),
    "davies_bouldin_score": (
        "Davies_Bouldin_Index",
        davies_bouldin_score,
        "min"
    ),
    "calinski_harabasz_score": (
        "Calinski_Harabasz_Score",
        calinski_harabasz_score,
        "max"
    )
}

FEATURE_SPACE_LABELS = {
    "without_gender": "Without_Gender",
    "with_gender": "With_Gender"
}


# ============================================================
# 4. PREPARE DIRECTORIES
# ============================================================

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 5. LOAD RAW DATA
# ============================================================

def load_data():

    print("\n========== LOADING RAW DATA ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = clean_records(pd.read_csv(DATA_PATH))

    if "Gender" not in df.columns:
        raise ValueError(
            "Gender column not found in dataset."
        )

    print("Dataset shape:", df.shape)

    return df


# ============================================================
# 6. DEFINE EXPERIMENT CONFIGURATIONS
# ============================================================

def get_experiment_configs():

    print("\n========== LOADING EXPERIMENT CONFIGURATIONS ==========")

    if not CONFIG_FEATURE_SPACES:
        raise ValueError(
            "Feature spaces are missing in config.yaml."
        )

    if not CONFIG_SCALERS:
        raise ValueError(
            "Scaler configurations are missing in config.yaml."
        )

    feature_spaces = {}

    for name, features in CONFIG_FEATURE_SPACES.items():

        label = FEATURE_SPACE_LABELS.get(name, name)

        feature_spaces[label] = features

    scalers = {}

    for scaler_name in CONFIG_SCALERS:

        if scaler_name not in SCALER_CLASSES:
            raise ValueError(
                f"Unsupported scaler: {scaler_name}"
            )

        scalers[scaler_name] = SCALER_CLASSES[scaler_name]

    print("Feature spaces:", list(feature_spaces.keys()))

    print("Scalers:", list(scalers.keys()))

    print("K values:", list(K_VALUES))

    expected = (
        len(feature_spaces)
        * len(scalers)
        * len(K_VALUES)
    )

    print("Expected experiments:", expected)

    return feature_spaces, scalers


# ============================================================
# 7. EVALUATION METRICS
# ============================================================

def calculate_metrics(X, labels):

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError(
            "Only one cluster was generated."
        )

    if len(unique_labels) >= len(X):
        raise ValueError(
            "Invalid clustering: number of clusters "
            "must be smaller than number of samples."
        )

    metrics = {}

    for metric_name in CONFIG_METRICS:

        if metric_name not in METRIC_REGISTRY:
            raise ValueError(
                f"Unsupported evaluation metric: {metric_name}"
            )

        column_name, metric_function, _ = (
            METRIC_REGISTRY[metric_name]
        )

        metrics[column_name] = metric_function(
            X, labels
        )

    return metrics


# ============================================================
# 8. RUN EXPERIMENTS
# ============================================================

def run_experiments(df):

    print("\n========== EXPERIMENT RUNNING ==========")

    feature_spaces, scalers = get_experiment_configs()

    results = []

    for feature_name, features in feature_spaces.items():

        missing_columns = [
            feature for feature in features
            if feature not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing features for {feature_name}: "
                f"{missing_columns}"
            )

        for scaler_name, scaler_class in scalers.items():

            print("\n--------------------------------")

            print("Feature Space:", feature_name)

            print("Scaler:", scaler_name)

            scaler = scaler_class()

            feature_frame = prepare_feature_frame(
                df, features, GENDER_MAPPING
            )
            X = scaler.fit_transform(feature_frame)

            for k in K_VALUES:

                model = KMeans(
                    n_clusters=k,
                    max_iter=MAX_ITER,
                    tol=TOLERANCE,
                    random_state=RANDOM_STATE,
                    init=INITIALIZATION
                )

                labels = model.fit_predict(X)

                metrics = calculate_metrics(
                    X, labels
                )

                result = {

                    "Feature_Space": feature_name,

                    "Scaler": scaler_name,

                    "K": k,

                    "Inertia": model.inertia_,

                    "Iterations": model.n_iter_,

                    "Actual_Clusters": len(
                        np.unique(labels)
                    ),

                    **metrics

                }

                results.append(result)

                print(
                    f"K={k} | "
                    f"Silhouette={metrics['Silhouette_Score']:.4f} | "
                    f"DBI={metrics['Davies_Bouldin_Index']:.4f} | "
                    f"CH={metrics['Calinski_Harabasz_Score']:.4f}"
                )

    results_df = pd.DataFrame(results)

    expected_count = (
        len(feature_spaces)
        * len(scalers)
        * len(K_VALUES)
    )

    if len(results_df) != expected_count:
        raise RuntimeError(
            f"Experiment count mismatch: "
            f"expected {expected_count}, "
            f"got {len(results_df)}."
        )

    print("\nTotal experiments completed:", len(results_df))

    return results_df


# ============================================================
# 9. SAVE RESULTS
# ============================================================

def save_results(results):

    output_path = (
        RESULT_DIR / "feature_scaling_experiments.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nResults saved:", output_path)


# ============================================================
# 10. PLOT COMPARISON
# ============================================================

def plot_comparison(results, metric, title, filename):

    plt.figure(figsize=(11, 7))

    for (feature, scaler), group in results.groupby(
        ["Feature_Space", "Scaler"]
    ):

        group = group.sort_values("K")

        label = f"{feature} - {scaler}"

        plt.plot(
            group["K"],
            group[metric],
            marker="o",
            label=label
        )

    plt.title(title)

    plt.xlabel("Number of Clusters (K)")

    plt.ylabel(metric)

    plt.xticks(list(K_VALUES))

    plt.grid(True)

    plt.legend()

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
# 11. SUMMARIZE BEST CONFIGURATIONS
# ============================================================

def summarize_results(results):

    print("\n========== BEST CONFIGURATIONS ==========")

    if results.empty:
        raise ValueError(
            "No experiment results available."
        )

    for metric_name in CONFIG_METRICS:

        column_name, _, direction = (
            METRIC_REGISTRY[metric_name]
        )

        if column_name not in results.columns:
            raise KeyError(
                f"Metric column not found: {column_name}"
            )

        if direction == "max":

            best = results.loc[
                results[column_name].idxmax()
            ]

        else:

            best = results.loc[
                results[column_name].idxmin()
            ]

        print(f"\nMetric: {column_name}")

        print("Feature Space:", best["Feature_Space"])

        print("Scaler:", best["Scaler"])

        print("K:", int(best["K"]))

        print("Score:", round(best[column_name], 4))


# ============================================================
# 12. MAIN
# ============================================================

def main():

    print("\n========================================")

    print("MALL CUSTOMER CLUSTERING EXPERIMENTS")

    print("========================================")

    df = load_data()

    results = run_experiments(df)

    save_results(results)

    plot_comparison(
        results,
        "Silhouette_Score",
        "Silhouette Comparison",
        "silhouette_experiments.png"
    )

    plot_comparison(
        results,
        "Davies_Bouldin_Index",
        "Davies-Bouldin Comparison",
        "dbi_experiments.png"
    )

    plot_comparison(
        results,
        "Calinski_Harabasz_Score",
        "Calinski-Harabasz Comparison",
        "ch_experiments.png"
    )

    summarize_results(results)

    print("\n========================================")

    print("EXPERIMENTS COMPLETED SUCCESSFULLY!")

    print("========================================")


if __name__ == "__main__":
    main()
