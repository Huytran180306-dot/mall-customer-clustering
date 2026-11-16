import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

# ==============================
# CONFIGURATION
# ==============================

ROOT_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT_DIR / "data" / "raw" / "Mall_Customers.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models" / "preprocessing"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ==============================
# LOAD DATA
# ==============================

def load_data():
    print("\nLOADING DATA...")

    df = pd.read_csv(DATA_PATH)

    print("Dataset shape:", df.shape)

    return df


# ==============================
# CLEAN DATA
# ==============================

def clean_data(df):
    print("\nCLEANING DATA...")

    df = df.copy()

    df.columns = df.columns.str.strip()

    df = df.drop_duplicates()

    df = df.dropna()

    print("Missing values:", df.isnull().sum().sum())
    print("Duplicates:", df.duplicated().sum())

    return df


# ==============================
# FEATURE SELECTION
# ==============================

def select_features(df):
    print("\nSELECTING FEATURES...")

    features = df[
        [
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ]
    ].copy()

    print("Selected features:", features.columns.tolist())

    return features


# ==============================
# ENCODING
# ==============================

def encode_gender(df):
    print("\nENCODING GENDER...")

    df = df.copy()

    df["Gender"] = df["Gender"].map({
        "Male": 0,
        "Female": 1
    })

    return df


# ==============================
# FEATURE SCALING
# ==============================

def scale_features(df):

    print("\nSCALING FEATURES...")

    standard_scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    standard_data = standard_scaler.fit_transform(df)

    minmax_data = minmax_scaler.fit_transform(df)

    standard_df = pd.DataFrame(
        standard_data,
        columns=df.columns
    )

    minmax_df = pd.DataFrame(
        minmax_data,
        columns=df.columns
    )

    joblib.dump(
        standard_scaler,
        MODEL_DIR / "standard_scaler.pkl"
    )

    joblib.dump(
        minmax_scaler,
        MODEL_DIR / "minmax_scaler.pkl"
    )

    return standard_df, minmax_df


# ==============================
# SAVE DATA
# ==============================

def save_data(standard_df, minmax_df):

    standard_df.to_csv(
        OUTPUT_DIR / "Mall_Customers_StandardScaler.csv",
        index=False
    )

    minmax_df.to_csv(
        OUTPUT_DIR / "Mall_Customers_MinMaxScaler.csv",
        index=False
    )

    print("\nDATA SAVED SUCCESSFULLY!")


# ==============================
# MAIN
# ==============================

def main():

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