"""
K-Means training module for Mall Customer Clustering.

Responsibilities:
1. Load standardized customer data.
2. Train custom K-Means for K values from configuration.
3. Record inertia and iteration counts.
4. Save Elbow Method results.
5. Generate the Elbow Method visualization.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.config import get_config, get_path
from src.models.kmeans import KMeans


# ============================================================
# 2. CENTRALIZED CONFIGURATION
# ============================================================

DATA_PATH = (
    get_path("data", "processed_dir")
    / "Mall_Customers_StandardScaler.csv"
)

RESULT_DIR = get_path("paths", "progress")

FIGURE_DIR = (
    get_path("paths", "figures")
    / "02_training"
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


# ============================================================
# 3. VALIDATE CONFIGURATION
# ============================================================

if not K_VALUES:
    raise ValueError(
        "K values are missing in config.yaml."
    )

if any(not isinstance(k, int) or k < 2 for k in K_VALUES):
    raise ValueError(
        "All K values must be integers greater than or equal to 2."
    )

if len(set(K_VALUES)) != len(K_VALUES):
    raise ValueError(
        "Duplicate K values detected in config.yaml."
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
            "Training dataset is empty."
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
# 6. TRAIN DIFFERENT K VALUES
# ============================================================

def run_kmeans_experiments(X):

    print("\n========== K-MEANS EXPERIMENTS ==========")

    results = []

    for k in K_VALUES:

        print(f"\nTraining K-Means with K = {k}")

        model = KMeans(
            n_clusters=k,
            max_iter=MAX_ITER,
            tol=TOLERANCE,
            random_state=RANDOM_STATE,
            init=INITIALIZATION
        )

        model.fit(X)

        results.append({

            "K": k,

            "Inertia": model.inertia_,

            "Iterations": model.n_iter_

        })

        print("Inertia:", model.inertia_)

        print("Iterations:", model.n_iter_)

    results_df = pd.DataFrame(results)

    if len(results_df) != len(K_VALUES):
        raise RuntimeError(
            "K-Means training experiment count mismatch."
        )

    return results_df


# ============================================================
# 7. SAVE EXPERIMENT RESULTS
# ============================================================

def save_results(results):

    output_path = (
        RESULT_DIR / "kmeans_elbow_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nResults saved:", output_path)


# ============================================================
# 8. ELBOW METHOD
# ============================================================

def plot_elbow(results):

    plt.figure(figsize=(9, 6))

    plt.plot(
        results["K"],
        results["Inertia"],
        marker="o"
    )

    plt.title("Elbow Method - K-Means")

    plt.xlabel("Number of Clusters (K)")

    plt.ylabel("Inertia")

    plt.xticks(list(K_VALUES))

    plt.grid(True)

    plt.tight_layout()

    output_path = (
        FIGURE_DIR / "elbow_method.png"
    )

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nElbow plot saved:", output_path)


# ============================================================
# 9. MAIN
# ============================================================

def main():

    print("\n================================")

    print("MALL CUSTOMER K-MEANS TRAINING")

    print("================================")

    print("K values:", list(K_VALUES))

    print("Random state:", RANDOM_STATE)

    print("Max iterations:", MAX_ITER)

    print("Tolerance:", TOLERANCE)

    X = load_data()

    results = run_kmeans_experiments(X)

    print("\n========== EXPERIMENT SUMMARY ==========")

    print(results.to_string(index=False))

    save_results(results)

    plot_elbow(results)

    print("\n================================")

    print("TRAINING EXPERIMENTS COMPLETED!")

    print("================================")

    print(
        "\nNote: Select optimal K by inspecting "
        "the Elbow plot and evaluating cluster quality."
    )


if __name__ == "__main__":
    main()