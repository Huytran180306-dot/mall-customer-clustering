from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

from src.config import get_config, get_path
from src.models.kmeans import KMeans


# ==========================================
# CONFIGURATION
# ==========================================

BEHAVIOR_CONFIG = get_config("behavior_analysis", default={})
MODEL_CONFIG = get_config("model", default={})

FEATURE_SPACE_NAME = BEHAVIOR_CONFIG.get(
    "feature_space",
    "without_gender"
)

FEATURES = get_config(
    "experiments",
    "feature_spaces",
    FEATURE_SPACE_NAME
)

if not FEATURES:
    raise ValueError(
        f"Feature space '{FEATURE_SPACE_NAME}' not found in config.yaml"
    )

SCALER_NAME = BEHAVIOR_CONFIG.get(
    "scaler",
    "StandardScaler"
)

SCALER_MAP = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler
}

if SCALER_NAME not in SCALER_MAP:
    raise ValueError(
        f"Unsupported scaler: {SCALER_NAME}"
    )

SCALER_CLASS = SCALER_MAP[SCALER_NAME]

DATA_PATH = get_path("data", "raw_path")

OUTPUT_DIR = (
    get_path("paths", "figures")
    / "05_extended"
    / "convergence"
)

RESULT_DIR = get_path("paths", "progress")

K = int(BEHAVIOR_CONFIG.get("n_clusters", 6))

MAX_ITER = int(
    MODEL_CONFIG.get("max_iterations", 300)
)

TOL = float(
    MODEL_CONFIG.get("tolerance", 0.0001)
)

RANDOM_STATE = int(
    MODEL_CONFIG.get("random_state", 42)
)

INITIALIZATION = MODEL_CONFIG.get(
    "initialization",
    "k-means++"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 1. LOAD DATA
# ==========================================

def load_data():

    print("\n========== LOADING DATA ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    missing_features = [
        feature for feature in FEATURES
        if feature not in df.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing features in dataset: {missing_features}"
        )

    X = df[FEATURES].to_numpy(dtype=float)

    if not np.isfinite(X).all():
        raise ValueError(
            "Dataset contains missing or infinite values."
        )

    scaler = SCALER_CLASS()

    X_scaled = scaler.fit_transform(X)

    print("Dataset shape:", X_scaled.shape)
    print("Features:", FEATURES)
    print("Scaler:", SCALER_NAME)

    return X_scaled


# ==========================================
# 2. TRAIN MODEL
# ==========================================

def train_model(X):

    print("\n========== TRAINING K-MEANS ==========")

    model = KMeans(
        n_clusters=K,
        max_iter=MAX_ITER,
        tol=TOL,
        random_state=RANDOM_STATE,
        init=INITIALIZATION
    )

    model.fit(X)

    print("Clusters:", K)
    print("Iterations:", model.n_iter_)
    print("Inertia:", model.inertia_)

    if not model.centroid_history_:
        raise RuntimeError(
            "Centroid history is empty."
        )

    print(
        "Centroid history steps:",
        len(model.centroid_history_)
    )

    return model


# ==========================================
# 3. PCA VISUALIZATION
# ==========================================

def visualize_convergence(X, model):

    print("\n========== PCA VISUALIZATION ==========")

    pca = PCA(n_components=2)

    X_2d = pca.fit_transform(X)

    history = np.asarray(
        model.centroid_history_
    )

    history_2d = pca.transform(
        history.reshape(-1, X.shape[1])
    ).reshape(
        history.shape[0],
        history.shape[1],
        2
    )

    final_labels = model.labels_

    plt.figure(figsize=(11, 8))

    # Customer points
    plt.scatter(
        X_2d[:, 0],
        X_2d[:, 1],
        c=final_labels,
        cmap="tab10",
        s=45,
        alpha=0.65,
        label="Customers"
    )

    # Centroid movement
    for cluster_id in range(K):

        path = history_2d[:, cluster_id, :]

        plt.plot(
            path[:, 0],
            path[:, 1],
            marker="o",
            markersize=5,
            linewidth=1.5,
            label=f"Centroid {cluster_id}"
        )

    # Initial centroids
    plt.scatter(
        history_2d[0, :, 0],
        history_2d[0, :, 1],
        marker="s",
        s=130,
        edgecolors="black",
        label="Initial Centroids"
    )

    # Final centroids
    plt.scatter(
        history_2d[-1, :, 0],
        history_2d[-1, :, 1],
        marker="X",
        s=220,
        edgecolors="black",
        label="Final Centroids"
    )

    plt.title(
        "K-Means Centroid Convergence - PCA Visualization"
    )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left"
    )

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = OUTPUT_DIR / "kmeans_convergence.png"

    plt.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("Figure saved:", output_path)


# ==========================================
# 4. SAVE CENTROID HISTORY
# ==========================================

def save_centroid_history(model):

    feature_column_map = {
        "Age": "Age_Scaled",
        "Annual Income (k$)": "Annual_Income_Scaled",
        "Spending Score (1-100)": "Spending_Score_Scaled",
        "Gender": "Gender_Scaled"
    }

    output_columns = [
        feature_column_map.get(
            feature,
            f"{feature.replace(' ', '_')}_Scaled"
        )
        for feature in FEATURES
    ]

    records = []

    for iteration, centroids in enumerate(
        model.centroid_history_
    ):

        for cluster_id, centroid in enumerate(centroids):

            record = {
                "Iteration": iteration,
                "Cluster": cluster_id
            }

            for feature, value in zip(FEATURES, centroid):

                column_name = feature_column_map.get(
                    feature,
                    f"{feature.replace(' ', '_')}_Scaled"
                )

                record[column_name] = value

            records.append(record)

    history_df = pd.DataFrame(
        records,
        columns=[
            "Iteration",
            "Cluster",
            *output_columns
        ]
    )

    output_path = RESULT_DIR / "centroid_history.csv"

    history_df.to_csv(
        output_path,
        index=False
    )

    print("Centroid history saved:", output_path)


# ==========================================
# 5. MAIN
# ==========================================

def main():

    print("\n========================================")
    print("K-MEANS CONVERGENCE DEMONSTRATION")
    print("========================================")

    print("Feature space:", FEATURE_SPACE_NAME)
    print("Features:", FEATURES)
    print("Scaler:", SCALER_NAME)
    print("K:", K)

    X = load_data()

    model = train_model(X)

    visualize_convergence(X, model)

    save_centroid_history(model)

    print("\n========================================")
    print("CONVERGENCE DEMO COMPLETED SUCCESSFULLY!")
    print("========================================")


if __name__ == "__main__":
    main()