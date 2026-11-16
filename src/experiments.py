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

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.kmeans import KMeans


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = ROOT_DIR / "data/raw/Mall_Customers.csv"

RESULT_DIR = ROOT_DIR / "progress_results"

FIGURE_DIR = (
    ROOT_DIR
    / "reports/figures/customer_segmentation/05_extended/experiments"
)

K_VALUES = range(2, 11)

RANDOM_STATE = 42
MAX_ITER = 300
TOLERANCE = 0.0001

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. LOAD RAW DATA
# ==========================================

def load_data():

    print("\n========== LOADING RAW DATA ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    df = df.drop_duplicates().dropna().copy()

    df["Gender"] = df["Gender"].map({
        "Male": 0,
        "Female": 1
    })

    if df["Gender"].isna().any():
        raise ValueError("Unexpected Gender values.")

    print("Dataset shape:", df.shape)

    return df


# ==========================================
# 2. DEFINE EXPERIMENTS
# ==========================================

def get_experiment_configs():

    feature_spaces = {

        "Without_Gender": [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ],

        "With_Gender": [
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]

    }

    scalers = {

        "StandardScaler": StandardScaler,

        "MinMaxScaler": MinMaxScaler

    }

    return feature_spaces, scalers


# ==========================================
# 3. EVALUATION METRICS
# ==========================================

def calculate_metrics(X, labels):

    unique_labels = np.unique(labels)

    if len(unique_labels) < 2:
        raise ValueError("Only one cluster was generated.")

    return {

        "Silhouette_Score": silhouette_score(
            X, labels
        ),

        "Davies_Bouldin_Index": davies_bouldin_score(
            X, labels
        ),

        "Calinski_Harabasz_Score": calinski_harabasz_score(
            X, labels
        )

    }


# ==========================================
# 4. RUN EXPERIMENTS
# ==========================================

def run_experiments(df):

    print("\n========== EXPERIMENT RUNNING ==========")

    feature_spaces, scalers = get_experiment_configs()

    results = []

    for feature_name, features in feature_spaces.items():

        for scaler_name, scaler_class in scalers.items():

            print("\n--------------------------------")

            print("Feature Space:", feature_name)
            print("Scaler:", scaler_name)

            scaler = scaler_class()

            X = scaler.fit_transform(
                df[features]
            )

            for k in K_VALUES:

                model = KMeans(
                    n_clusters=k,
                    max_iter=MAX_ITER,
                    tol=TOLERANCE,
                    random_state=RANDOM_STATE,
                    init="k-means++"
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

                    "Actual_Clusters": len(np.unique(labels)),

                    **metrics

                }

                results.append(result)

                print(
                    f"K={k} | "
                    f"Silhouette={metrics['Silhouette_Score']:.4f} | "
                    f"DBI={metrics['Davies_Bouldin_Index']:.4f} | "
                    f"CH={metrics['Calinski_Harabasz_Score']:.4f}"
                )

    return pd.DataFrame(results)


# ==========================================
# 5. SAVE RESULTS
# ==========================================

def save_results(results):

    output_path = (
        RESULT_DIR / "feature_scaling_experiments.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nResults saved:", output_path)


# ==========================================
# 6. PLOT COMPARISON
# ==========================================

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


# ==========================================
# 7. BEST CONFIGURATION
# ==========================================

def summarize_results(results):

    print("\n========== BEST CONFIGURATIONS ==========")

    metrics = {

        "Silhouette_Score": "max",

        "Davies_Bouldin_Index": "min",

        "Calinski_Harabasz_Score": "max"

    }

    for metric, direction in metrics.items():

        if direction == "max":
            best = results.loc[
                results[metric].idxmax()
            ]

        else:
            best = results.loc[
                results[metric].idxmin()
            ]

        print(f"\nMetric: {metric}")

        print("Feature Space:", best["Feature_Space"])

        print("Scaler:", best["Scaler"])

        print("K:", int(best["K"]))

        print("Score:", round(best[metric], 4))


# ==========================================
# 8. MAIN
# ==========================================

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