import sys
from pathlib import Path

import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.models.kmeans import KMeans


# ==========================================
# CONFIGURATION
# ==========================================

DATA_PATH = ROOT_DIR / "data/raw/Mall_Customers.csv"

RESULT_DIR = ROOT_DIR / "progress_results"

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

def load_data():

    df = pd.read_csv(DATA_PATH)

    df = df.drop_duplicates().dropna().copy()

    return df


# ==========================================
# ANALYZE CLUSTER PROFILES
# ==========================================

def analyze_configuration(
    df,
    configuration_name,
    features,
    scaler,
    n_clusters
):

    X = df[features].copy()

    X["Gender"] = X["Gender"].map({
        "Male": 0,
        "Female": 1
    }) if "Gender" in X.columns else None

    X = X.drop(
        columns=["Gender"],
        errors="ignore"
    ) if "Gender" not in features else X

    if "Gender" in features:
        X["Gender"] = df["Gender"].map({
            "Male": 0,
            "Female": 1
        })

    scaled_data = scaler.fit_transform(X)

    model = KMeans(
        n_clusters=n_clusters,
        max_iter=300,
        random_state=42,
        init="k-means++"
    )

    labels = model.fit_predict(scaled_data)

    result = df.copy()

    result["Cluster"] = labels

    # Cluster sizes
    print("\n========== CLUSTER SIZES ==========")

    print(
        result["Cluster"].value_counts().sort_index()
    )

    # Cluster profiles
    profile = result.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean")
    ).round(2)

    print("\n========== CLUSTER PROFILES ==========")

    print(profile)

    # Gender distribution
    gender_profile = pd.crosstab(
        result["Cluster"],
        result["Gender"],
        normalize="index"
    ).round(3)

    print("\n========== GENDER DISTRIBUTION ==========")

    print(gender_profile)

    # Save results
    profile.to_csv(
        RESULT_DIR / f"{configuration_name}_profile.csv"
    )

    gender_profile.to_csv(
        RESULT_DIR / f"{configuration_name}_gender.csv"
    )

    result.to_csv(
        RESULT_DIR / f"{configuration_name}_customers.csv",
        index=False
    )

    print("\nSaved configuration:", configuration_name)


# ==========================================
# MAIN
# ==========================================

def main():

    print("\n================================")

    print("CUSTOMER CLUSTER PROFILE ANALYSIS")

    print("================================")

    df = load_data()

    # Configuration A:
    # Without Gender + StandardScaler + K=6

    analyze_configuration(
        df,
        "without_gender_standard_k6",
        [
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ],
        StandardScaler(),
        6
    )

    # Configuration B:
    # With Gender + MinMaxScaler + K=2

    analyze_configuration(
        df,
        "with_gender_minmax_k2",
        [
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)"
        ],
        MinMaxScaler(),
        2
    )

    print("\n================================")

    print("CLUSTER ANALYSIS COMPLETED!")

    print("================================")


if __name__ == "__main__":
    main()