"""
K-Means training module for Mall Customer Clustering.

Responsibilities:
1. Load raw customer data.
2. Select features from final_model configuration.
3. Encode categorical features when required.
4. Apply the configured scaler.
5. Train custom K-Means for different K values.
6. Record inertia and iteration counts.
7. Save Elbow Method results and visualization.
"""

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler


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
    / "02_training"
)

FINAL_CONFIG = get_config("final_model")

FEATURE_SPACE = FINAL_CONFIG["feature_space"]

FEATURE_COLUMNS = get_config(
    "experiments",
    "feature_spaces",
    FEATURE_SPACE
)

SCALER_NAME = FINAL_CONFIG["scaler"]

GENDER_MAPPING = get_config(
    "preprocessing",
    "encoding",
    "mapping",
    default={}
)

MODEL_DEFAULTS = get_config("model")

SCALER_CLASSES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler
}

K_VALUES = get_config("experiments", "k_values")

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


# ============================================================
# 3. VALIDATE CONFIGURATION
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

if FEATURE_SPACE not in ("with_gender", "without_gender"):
    raise ValueError(
        f"Unsupported feature space: {FEATURE_SPACE}"
    )

if SCALER_NAME not in SCALER_CLASSES:
    raise ValueError(
        f"Unsupported scaler: {SCALER_NAME}"
    )

if not FEATURE_COLUMNS:
    raise ValueError(
        "Feature columns are missing in config.yaml."
    )


# ============================================================
# 4. PREPARE DIRECTORIES
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
# 5. LOAD AND PREPARE DATA
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

    # Apply configured scaler
    scaler = SCALER_CLASSES[SCALER_NAME]()

    X_scaled = scaler.fit_transform(X)

    print("Scaling completed successfully.")

    return X_scaled


# ============================================================
# 6. TRAIN DIFFERENT K VALUES
# ============================================================

def run_kmeans_experiments(X):

    print("\n========== K-MEANS EXPERIMENTS ==========")

    results = []

    for k in K_VALUES:

        if k > len(X):
            raise ValueError(
                f"K={k} exceeds the number of samples."
            )

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

        print("Inertia:", round(model.inertia_, 6))
        print("Iterations:", model.n_iter_)

    results_df = pd.DataFrame(results)

    if len(results_df) != len(K_VALUES):
        raise RuntimeError(
            "K-Means experiment count mismatch."
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

    plt.title(
        f"Elbow Method - {FEATURE_SPACE} - {SCALER_NAME}"
    )

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

    print("\n========================================")

    print("MALL CUSTOMER K-MEANS TRAINING")

    print("========================================")

    print("Feature space:", FEATURE_SPACE)
    print("Features:", FEATURE_COLUMNS)
    print("Scaler:", SCALER_NAME)
    print("K values:", list(K_VALUES))
    print("Random state:", RANDOM_STATE)
    print("Max iterations:", MAX_ITER)
    print("Tolerance:", TOLERANCE)

    X = load_data()

    results = run_kmeans_experiments(X)

    print("\n========== EXPERIMENT SUMMARY ==========")

    print(
        results.to_string(index=False)
    )

    save_results(results)

    plot_elbow(results)

    print("\n========================================")

    print("TRAINING EXPERIMENTS COMPLETED!")

    print("========================================")


if __name__ == "__main__":
    main()
