"""
Final K-Means model training module.

Responsibilities:
1. Load original customer data.
2. Fit the configured scaler on all training samples.
3. Train the final K-Means model.
4. Save model, fitted scaler and metadata.
"""

import sys
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler


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

FINAL_CONFIG = get_config("final_model")

DATA_PATH = get_path("data", "raw_path")

MODEL_DIR = get_path("paths", "kmeans_models")

FEATURE_SPACE = FINAL_CONFIG["feature_space"]

FEATURE_COLUMNS = get_config(
    "experiments",
    "feature_spaces",
    FEATURE_SPACE
)

SCALER_NAME = FINAL_CONFIG["scaler"]

SCALER_CLASSES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler
}

SCALER_FILE_NAMES = {
    "StandardScaler": "standard_scaler.pkl",
    "MinMaxScaler": "minmax_scaler.pkl"
}

if SCALER_NAME not in SCALER_CLASSES:
    raise ValueError(
        f"Unsupported scaler: {SCALER_NAME}"
    )

N_CLUSTERS = int(FINAL_CONFIG["n_clusters"])

MAX_ITER = int(FINAL_CONFIG["max_iterations"])

TOLERANCE = float(FINAL_CONFIG["tolerance"])

RANDOM_STATE = int(FINAL_CONFIG["random_state"])

INITIALIZATION = FINAL_CONFIG["initialization"]

GENDER_MAPPING = get_config(
    "preprocessing", "encoding", "mapping"
)

MODEL_PATH = MODEL_DIR / "mall_customer_kmeans.pkl"

SCALER_OUTPUT = (
    MODEL_DIR / SCALER_FILE_NAMES[SCALER_NAME]
)

METADATA_PATH = MODEL_DIR / "model_metadata.json"


# ============================================================
# 3. VALIDATE CONFIGURATION
# ============================================================

if not FEATURE_COLUMNS:
    raise ValueError("Feature columns are missing.")

if N_CLUSTERS < 2:
    raise ValueError("K must be at least 2.")


# ============================================================
# 4. LOAD DATA
# ============================================================

def load_data():

    print("\nLOADING CUSTOMER DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    missing_columns = [
        column for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing features: {missing_columns}"
        )

    X = df[FEATURE_COLUMNS].copy()

    if "Gender" in X.columns:

        gender_values = (
            X["Gender"]
            .astype(str)
            .str.strip()
            .str.title()
        )

        X["Gender"] = gender_values.map(
            GENDER_MAPPING
        )

    for column in X.columns:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError(
            "Dataset contains missing or invalid features."
        )

    print("Dataset shape:", X.shape)
    print("Features:", FEATURE_COLUMNS)

    return X


# ============================================================
# 5. TRAIN FINAL MODEL
# ============================================================

def train_model(X):

    print("\nFITTING SCALER ON FULL TRAINING DATA")

    scaler = SCALER_CLASSES[SCALER_NAME]()

    # Important: fit scaler on all 200 original customers.
    X_scaled = scaler.fit_transform(X)

    print("Scaler:", SCALER_NAME)
    print("Scaler fitted successfully.")

    print("\nTRAINING FINAL K-MEANS MODEL")

    model = KMeans(
        n_clusters=N_CLUSTERS,
        max_iter=MAX_ITER,
        tol=TOLERANCE,
        random_state=RANDOM_STATE,
        init=INITIALIZATION
    )

    labels = model.fit_predict(X_scaled)

    print("Training completed!")

    print("Clusters:", sorted(set(labels)))

    print("Iterations:", model.n_iter_)

    print("Inertia:", model.inertia_)

    return model, scaler, labels


# ============================================================
# 6. SAVE MODEL ARTIFACTS
# ============================================================

def save_model(model, scaler, labels):

    print("\nSAVING MODEL")

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(model, MODEL_PATH)

    joblib.dump(scaler, SCALER_OUTPUT)

    cluster_sizes = (
        pd.Series(labels)
        .value_counts()
        .sort_index()
        .to_dict()
    )

    configuration_name = (
        f"{FEATURE_SPACE}_"
        f"{SCALER_NAME.replace('Scaler', '').lower()}_"
        f"k{N_CLUSTERS}"
    )

    metadata = {

        "model_name": "Mall Customer K-Means",

        "algorithm": "K-Means",

        "configuration": configuration_name,

        "n_clusters": N_CLUSTERS,

        "max_iter": MAX_ITER,

        "tolerance": TOLERANCE,

        "random_state": RANDOM_STATE,

        "initialization": INITIALIZATION,

        "scaler": SCALER_NAME,

        "features": FEATURE_COLUMNS,

        "gender_mapping": GENDER_MAPPING,

        "training_samples": len(labels),

        "inertia": float(model.inertia_),

        "iterations": int(model.n_iter_),

        "cluster_sizes": {
            str(k): int(v)
            for k, v in cluster_sizes.items()
        }

    }

    with open(
        METADATA_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nMODEL SAVED SUCCESSFULLY!")

    print("Model:", MODEL_PATH)

    print("Scaler:", SCALER_OUTPUT)

    print("Metadata:", METADATA_PATH)


# ============================================================
# 7. MAIN
# ============================================================

def main():

    print("=" * 55)

    print("MALL CUSTOMER FINAL MODEL TRAINING")

    print("=" * 55)

    print("Feature space:", FEATURE_SPACE)

    print("Scaler:", SCALER_NAME)

    print("Number of clusters:", N_CLUSTERS)

    print("Random state:", RANDOM_STATE)

    X = load_data()

    model, scaler, labels = train_model(X)

    save_model(model, scaler, labels)

    print("\nFINAL MODEL READY!")


if __name__ == "__main__":
    main()