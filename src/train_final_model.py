import json
import sys
from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.models.kmeans import KMeans


DATA_PATH = ROOT_DIR / "data/raw/Mall_Customers.csv"

SCALER_PATH = (
    ROOT_DIR / "models/preprocessing/minmax_scaler.pkl"
)

MODEL_DIR = ROOT_DIR / "models/kmeans"

MODEL_PATH = MODEL_DIR / "mall_customer_kmeans.pkl"

SCALER_OUTPUT = MODEL_DIR / "minmax_scaler.pkl"

METADATA_PATH = MODEL_DIR / "model_metadata.json"


FEATURE_COLUMNS = [
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

GENDER_MAPPING = {
    "Male": 0,
    "Female": 1
}


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    print("\nLOADING CUSTOMER DATA")

    if not DATA_PATH.exists():
        raise FileNotFoundError(DATA_PATH)

    df = pd.read_csv(DATA_PATH)

    X = df[FEATURE_COLUMNS].copy()

    X["Gender"] = (
        X["Gender"]
        .astype(str)
        .str.strip()
        .str.title()
        .map(GENDER_MAPPING)
    )

    if X.isnull().any().any():
        raise ValueError(
            "Dataset contains invalid or missing features."
        )

    print("Dataset shape:", X.shape)

    return X


# ==========================================
# 3. LOAD SCALER
# ==========================================

def load_scaler():

    print("\nLOADING MINMAX SCALER")

    if not SCALER_PATH.exists():
        raise FileNotFoundError(SCALER_PATH)

    scaler = joblib.load(SCALER_PATH)

    scaler_features = getattr(
        scaler,
        "feature_names_in_",
        None
    )

    if scaler_features is not None:
        if list(scaler_features) != FEATURE_COLUMNS:
            raise ValueError(
                "Scaler feature order does not match."
            )

    return scaler


# ==========================================
# 4. TRAIN FINAL MODEL
# ==========================================

def train_model(X, scaler):

    print("\nTRAINING FINAL K-MEANS MODEL")

    X_scaled = scaler.transform(X)

    model = KMeans(
        n_clusters=2,
        max_iter=300,
        random_state=42,
        init="k-means++"
    )

    labels = model.fit_predict(X_scaled)

    print("Training completed!")

    print("Clusters:", sorted(set(labels)))

    print("Iterations:", model.n_iter_)

    return model, labels


# ==========================================
# 5. SAVE MODEL ARTIFACTS
# ==========================================

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

    metadata = {
        "model_name": "Mall Customer K-Means",

        "algorithm": "K-Means",

        "configuration": "with_gender_minmax_k2",

        "n_clusters": 2,

        "max_iter": 300,

        "random_state": 42,

        "scaler": "MinMaxScaler",

        "features": FEATURE_COLUMNS,

        "gender_mapping": GENDER_MAPPING,

        "training_samples": len(labels),

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


# ==========================================
# 6. MAIN
# ==========================================

def main():

    print("=" * 50)

    print("MALL CUSTOMER FINAL MODEL TRAINING")

    print("=" * 50)

    X = load_data()

    scaler = load_scaler()

    model, labels = train_model(X, scaler)

    save_model(model, scaler, labels)

    print("\nFINAL MODEL READY!")


if __name__ == "__main__":

    main()