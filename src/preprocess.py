"""
Data preprocessing module for Mall Customer Clustering.

Responsibilities:
1. Load raw customer data.
2. Clean missing and duplicated records.
3. Select customer features.
4. Encode Gender.
5. Apply StandardScaler and MinMaxScaler.
6. Save processed datasets and scaler objects.
"""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.config import get_config, get_path


# ============================================================
# 1. CONFIGURATION
# ============================================================

DATA_PATH = get_path("data", "raw_path")
OUTPUT_DIR = get_path("data", "processed_dir")
MODEL_DIR = get_path("paths", "preprocessing_models")

FEATURE_COLUMNS = get_config(
    "data", "selected_features"
)

GENDER_MAPPING = get_config(
    "preprocessing", "encoding", "mapping"
)

HANDLE_MISSING = get_config(
    "preprocessing", "handle_missing_values", default=True
)

DROP_DUPLICATES = get_config(
    "preprocessing", "drop_duplicates", default=True
)

SCALER_NAMES = get_config(
    "preprocessing", "scalers",
    default=["StandardScaler", "MinMaxScaler"]
)

SCALER_CLASSES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler,
}

SCALER_FILE_NAMES = {
    "StandardScaler": "standard_scaler.pkl",
    "MinMaxScaler": "minmax_scaler.pkl",
}


# ============================================================
# 2. PREPARE DIRECTORIES
# ============================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    print("\nLOADING DATA...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    return df


# ============================================================
# 4. CLEAN DATA
# ============================================================

def clean_data(df):

    print("\nCLEANING DATA...")

    df = df.copy()

    df.columns = df.columns.str.strip()

    if DROP_DUPLICATES:
        before = len(df)
        df = df.drop_duplicates()
        print("Duplicates removed:", before - len(df))

    if HANDLE_MISSING:
        before = len(df)
        df = df.dropna()
        print("Missing records removed:", before - len(df))

    print("Missing values remaining:", df.isnull().sum().sum())
    print("Duplicates remaining:", df.duplicated().sum())

    return df


# ============================================================
# 5. FEATURE SELECTION
# ============================================================

def select_features(df):

    print("\nSELECTING FEATURES...")

    if not FEATURE_COLUMNS:
        raise ValueError(
            "No feature columns configured in config.yaml."
        )

    missing_columns = [
        col for col in FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing feature columns: {missing_columns}"
        )

    features = df[FEATURE_COLUMNS].copy()

    print("Selected features:", features.columns.tolist())

    return features


# ============================================================
# 6. GENDER ENCODING
# ============================================================

def encode_gender(df):

    print("\nENCODING GENDER...")

    df = df.copy()

    if "Gender" not in df.columns:
        return df

    if not GENDER_MAPPING:
        raise ValueError(
            "Gender mapping is missing in config.yaml."
        )

    gender_values = df["Gender"].astype(str).str.strip()

    encoded = gender_values.map(GENDER_MAPPING)

    if encoded.isna().any():
        unknown_values = gender_values[encoded.isna()].unique()

        raise ValueError(
            f"Unknown Gender values: {unknown_values.tolist()}"
        )

    df["Gender"] = encoded.astype(int)

    print("Gender mapping:", GENDER_MAPPING)

    return df


# ============================================================
# 7. FEATURE SCALING
# ============================================================

def scale_features(df):

    print("\nSCALING FEATURES...")

    scaled_results = {}

    for scaler_name in SCALER_NAMES:

        if scaler_name not in SCALER_CLASSES:
            raise ValueError(
                f"Unsupported scaler: {scaler_name}"
            )

        scaler = SCALER_CLASSES[scaler_name]()

        scaled_data = scaler.fit_transform(df)

        scaled_df = pd.DataFrame(
            scaled_data,
            columns=df.columns,
            index=df.index
        )

        file_name = SCALER_FILE_NAMES[scaler_name]

        joblib.dump(
            scaler,
            MODEL_DIR / file_name
        )

        scaled_results[scaler_name] = scaled_df

        print(f"{scaler_name} completed.")
        print(f"Scaler saved: {MODEL_DIR / file_name}")

    required_scalers = [
        "StandardScaler",
        "MinMaxScaler"
    ]

    for name in required_scalers:
        if name not in scaled_results:
            raise ValueError(
                f"Required scaler missing: {name}"
            )

    return (
        scaled_results["StandardScaler"],
        scaled_results["MinMaxScaler"]
    )


# ============================================================
# 8. SAVE PROCESSED DATA
# ============================================================

def save_data(standard_df, minmax_df):

    print("\nSAVING PROCESSED DATA...")

    standard_path = (
        OUTPUT_DIR / "Mall_Customers_StandardScaler.csv"
    )

    minmax_path = (
        OUTPUT_DIR / "Mall_Customers_MinMaxScaler.csv"
    )

    standard_df.to_csv(
        standard_path,
        index=False
    )

    minmax_df.to_csv(
        minmax_path,
        index=False
    )

    print("StandardScaler saved:", standard_path)
    print("MinMaxScaler saved:", minmax_path)


# ============================================================
# 9. MAIN PIPELINE
# ============================================================

def main():

    print("\n" + "=" * 55)
    print("MALL CUSTOMER DATA PREPROCESSING")
    print("=" * 55)

    df = load_data()

    df = clean_data(df)

    features = select_features(df)

    encoded = encode_gender(features)

    standard_df, minmax_df = scale_features(encoded)

    save_data(standard_df, minmax_df)

    print("\nPREPROCESSING COMPLETED SUCCESSFULLY!")

    print("\nStandardScaler preview:")
    print(standard_df.head())

    print("\nMinMaxScaler preview:")
    print(minmax_df.head())


if __name__ == "__main__":
    main()