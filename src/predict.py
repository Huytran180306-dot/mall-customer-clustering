"""Predict the cluster assignment for a new mall customer.

The model, scaler and feature columns are selected from config.yaml
(final_model), so prediction stays aligned with the trained model.
"""

import sys
import json
import math
from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Register the project's custom KMeans class before loading its pickle.
from src.models.kmeans import KMeans  # noqa: F401
from src.config import get_config, get_path
from src.feature_processing import prepare_feature_frame


FINAL_CONFIG = get_config("final_model")
FEATURE_SPACE = FINAL_CONFIG["feature_space"]
SCALER_NAME = FINAL_CONFIG["scaler"]
N_CLUSTERS = int(FINAL_CONFIG["n_clusters"])

FEATURE_COLUMNS = get_config(
    "experiments",
    "feature_spaces",
    FEATURE_SPACE
)

if not isinstance(FEATURE_COLUMNS, list) or not FEATURE_COLUMNS:
    raise ValueError(
        f"Invalid feature list for feature space: {FEATURE_SPACE}"
    )

SCALER_FILE_NAMES = {
    "StandardScaler": "standard_scaler.pkl",
    "MinMaxScaler": "minmax_scaler.pkl",
}

if SCALER_NAME not in SCALER_FILE_NAMES:
    raise ValueError(f"Unsupported scaler in config.yaml: {SCALER_NAME}")

MODEL_DIR = get_path("paths", "kmeans_models")
MODEL_PATH = MODEL_DIR / "mall_customer_kmeans.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
SCALER_PATH = MODEL_DIR / SCALER_FILE_NAMES[SCALER_NAME]

GENDER_MAPPING = get_config(
    "preprocessing",
    "encoding",
    "mapping",
    default={"Male": 0, "Female": 1}
)

# Normalize mapping keys so inputs such as "male" and "MALE" are accepted.
GENDER_MAPPING_NORMALIZED = {
    str(key).strip().lower(): int(value)
    for key, value in GENDER_MAPPING.items()
}


# ============================================================
# 2. LOAD TRAINED MODEL AND SCALER
# ============================================================

def load_model():
    print("\n========== LOADING TRAINED MODEL ==========")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}\n"
            "Run `python -m src.train_final_model` first."
        )

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler not found: {SCALER_PATH}\n"
            "Run `python -m src.train_final_model` first."
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {METADATA_PATH}\n"
            "Retrain the final model before predicting."
        )

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))

    print("Model loaded:", MODEL_PATH.name)
    print("Scaler loaded:", SCALER_PATH.name)
    print("Feature space:", FEATURE_SPACE)
    print("Features:", FEATURE_COLUMNS)
    print("Scaler type:", SCALER_NAME)
    print("Configured clusters:", N_CLUSTERS)

    # Confirm that the loaded scaler expects the configured number of columns.
    expected_features = getattr(scaler, "n_features_in_", None)
    if expected_features is not None and expected_features != len(FEATURE_COLUMNS):
        raise ValueError(
            "The saved scaler does not match config.yaml: "
            f"it expects {expected_features} features, but the current "
            f"configuration specifies {len(FEATURE_COLUMNS)}. "
            "Retrain the final model with `python -m src.train_final_model`."
        )

    centers = getattr(model, "cluster_centers_", None)
    if (
        metadata.get("features") != FEATURE_COLUMNS
        or metadata.get("scaler") != SCALER_NAME
        or int(metadata.get("n_clusters", -1)) != N_CLUSTERS
        or centers is None
        or centers.shape != (N_CLUSTERS, len(FEATURE_COLUMNS))
    ):
        raise ValueError(
            "Saved model, scaler and config.yaml do not match. "
            "Run `python -m src.train_final_model` to regenerate model artifacts."
        )

    scaler_features = getattr(scaler, "feature_names_in_", None)
    if scaler_features is not None and list(scaler_features) != FEATURE_COLUMNS:
        raise ValueError(
            "Saved scaler feature order does not match config.yaml. "
            "Retrain the final model before predicting."
        )

    print("MODEL AND SCALER LOADED SUCCESSFULLY!")
    return model, scaler


# ============================================================
# 3. INPUT VALIDATION HELPERS
# ============================================================

def read_number(prompt, minimum=None, maximum=None):
    """Read a numeric value and validate optional bounds."""
    while True:
        raw_value = input(prompt).strip()

        try:
            value = float(raw_value)
        except ValueError:
            print("Please enter a valid number.")
            continue

        if not math.isfinite(value):
            print("Please enter a finite number.")
            continue

        if minimum is not None and value < minimum:
            print(f"Value must be at least {minimum}.")
            continue

        if maximum is not None and value > maximum:
            print(f"Value must not exceed {maximum}.")
            continue

        return value


def read_gender():
    """Read and encode gender using the mapping in config.yaml."""
    valid_labels = list(GENDER_MAPPING.keys())
    choices = "/".join(str(label) for label in valid_labels)

    while True:
        value = input(f"Gender ({choices}): ").strip().lower()

        if value in GENDER_MAPPING_NORMALIZED:
            return GENDER_MAPPING_NORMALIZED[value]

        print(f"Invalid gender. Please enter one of: {choices}.")


# ============================================================
# 4. GET CUSTOMER INFORMATION
# ============================================================

def get_customer_input():
    print("\n========== ENTER CUSTOMER INFORMATION ==========")
    customer_values = {}

    # Ask for Gender only when it is part of the configured feature space.
    if "Gender" in FEATURE_COLUMNS:
        customer_values["Gender"] = read_gender()

    if "Age" in FEATURE_COLUMNS:
        customer_values["Age"] = read_number(
            "Age (years): ", minimum=1, maximum=120
        )

    if "Annual Income (k$)" in FEATURE_COLUMNS:
        customer_values["Annual Income (k$)"] = read_number(
            "Annual Income (k$): ", minimum=0
        )

    if "Spending Score (1-100)" in FEATURE_COLUMNS:
        customer_values["Spending Score (1-100)"] = read_number(
            "Spending Score (1-100): ", minimum=1, maximum=100
        )

    # Preserve exactly the same feature order used during model training.
    missing_features = [
        column for column in FEATURE_COLUMNS
        if column not in customer_values
    ]
    if missing_features:
        raise ValueError(
            "No input handler is defined for configured features: "
            f"{missing_features}"
        )

    customer = pd.DataFrame(
        [[customer_values[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS
    )

    return customer


# ============================================================
# 5. PREDICT CUSTOMER CLUSTER
# ============================================================

def predict_customer(model, scaler, customer):
    print("\nPROCESSING CUSTOMER DATA...")

    # Check feature order and column names before scaling.
    if list(customer.columns) != FEATURE_COLUMNS:
        raise ValueError(
            "Customer input columns do not match the model configuration. "
            f"Expected {FEATURE_COLUMNS}, received {list(customer.columns)}."
        )

    customer = prepare_feature_frame(
        customer, FEATURE_COLUMNS, GENDER_MAPPING
    )

    scaled_customer = scaler.transform(customer)
    cluster = model.predict(scaled_customer)

    return int(cluster[0])


# ============================================================
# 6. MAIN
# ============================================================

def main():
    print("=" * 52)
    print("MALL CUSTOMER CLUSTER PREDICTION")
    print("=" * 52)

    model, scaler = load_model()
    customer = get_customer_input()
    cluster = predict_customer(model, scaler, customer)

    print("\n" + "=" * 52)
    print("PREDICTION RESULT")
    print("=" * 52)
    print("\nCustomer Information:")
    print(customer.to_string(index=False))
    print("\nPredicted Cluster:", cluster)
    print("PREDICTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
