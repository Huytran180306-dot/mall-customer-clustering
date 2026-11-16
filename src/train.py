import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import custom K-Means
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.kmeans import KMeans


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = ROOT_DIR / "data/processed/Mall_Customers_StandardScaler.csv"

FIGURE_DIR = (
    ROOT_DIR
    / "reports/figures/customer_segmentation/02_training"
)

RESULT_DIR = ROOT_DIR / "progress_results"

RANDOM_STATE = 42
MAX_ITER = 300
TOLERANCE = 0.0001

K_VALUES = range(2, 11)


# ==========================================
# PREPARE DIRECTORIES
# ==========================================

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
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
# TRAIN DIFFERENT K VALUES
# ==========================================

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
            init="k-means++"
        )

        model.fit(X)

        results.append({
            "K": k,
            "Inertia": model.inertia_,
            "Iterations": model.n_iter_
        })

        print("Inertia:", model.inertia_)
        print("Iterations:", model.n_iter_)

    return pd.DataFrame(results)


# ==========================================
# SAVE EXPERIMENT RESULTS
# ==========================================

def save_results(results):

    output_path = (
        RESULT_DIR / "kmeans_elbow_results.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    print("\nResults saved:", output_path)


# ==========================================
# ELBOW METHOD
# ==========================================

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


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n================================")
    print("MALL CUSTOMER K-MEANS TRAINING")
    print("================================")

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
