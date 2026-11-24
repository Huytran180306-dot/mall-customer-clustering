"""
Customer cluster analysis module.

Responsibilities:
1. Load raw customer data.
2. Read behavioral-analysis and final-model configurations.
3. Generate customer cluster profiles.
4. Analyze gender distribution.
5. Save customer assignments and cluster summaries.
"""

import sys
from pathlib import Path

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
from src.feature_processing import clean_records, prepare_feature_frame


# ============================================================
# 2. CENTRALIZED CONFIGURATION
# ============================================================

DATA_PATH = get_path("data", "raw_path")
RESULT_DIR = get_path("paths", "progress")

FEATURE_SPACES = get_config(
    "experiments", "feature_spaces"
)

GENDER_MAPPING = get_config(
    "preprocessing", "encoding", "mapping"
)

MODEL_DEFAULTS = get_config("model")

SCALER_CLASSES = {
    "StandardScaler": StandardScaler,
    "MinMaxScaler": MinMaxScaler
}

RESULT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    print("\n========== LOADING RAW DATA ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = clean_records(pd.read_csv(DATA_PATH))

    required_columns = [
        "CustomerID",
        "Gender",
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Dataset shape:", df.shape)

    return df


# ============================================================
# 4. PREPARE FEATURE DATA
# ============================================================

def prepare_features(df, feature_space):

    if feature_space not in FEATURE_SPACES:
        raise ValueError(
            f"Unknown feature space: {feature_space}"
        )

    features = FEATURE_SPACES[feature_space]

    return prepare_feature_frame(df, features, GENDER_MAPPING)


# ============================================================
# 5. ANALYZE CLUSTER CONFIGURATION
# ============================================================

def analyze_configuration(
    df,
    configuration_name,
    configuration
):

    print("\n========================================")

    print("CONFIGURATION:", configuration_name)

    print("========================================")

    # Read configuration
    feature_space = configuration["feature_space"]

    scaler_name = configuration["scaler"]

    n_clusters = int(configuration["n_clusters"])

    max_iter = int(
        configuration.get(
            "max_iterations",
            MODEL_DEFAULTS["max_iterations"]
        )
    )

    tolerance = float(
        configuration.get(
            "tolerance",
            MODEL_DEFAULTS["tolerance"]
        )
    )

    random_state = int(
        configuration.get(
            "random_state",
            MODEL_DEFAULTS["random_state"]
        )
    )

    initialization = configuration.get(
        "initialization",
        MODEL_DEFAULTS["initialization"]
    )

    # Validate scaler
    if scaler_name not in SCALER_CLASSES:
        raise ValueError(
            f"Unsupported scaler: {scaler_name}"
        )

    # Prepare features
    X = prepare_features(df, feature_space)

    print("Feature space:", feature_space)
    print("Features:", X.columns.tolist())
    print("Scaler:", scaler_name)
    print("K:", n_clusters)

    # Scaling
    scaler = SCALER_CLASSES[scaler_name]()

    scaled_data = scaler.fit_transform(X)

    # K-Means
    model = KMeans(
        n_clusters=n_clusters,
        max_iter=max_iter,
        tol=tolerance,
        random_state=random_state,
        init=initialization
    )

    labels = model.fit_predict(scaled_data)

    # Attach cluster labels to original data
    result = df.copy()

    result["Cluster"] = labels

    # --------------------------------------------------------
    # Cluster sizes
    # --------------------------------------------------------

    print("\n========== CLUSTER SIZES ==========")

    print(
        result["Cluster"]
        .value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Cluster profiles
    # --------------------------------------------------------

    profile = result.groupby("Cluster").agg(
        Customers=("CustomerID", "count"),
        Average_Age=("Age", "mean"),
        Average_Income=("Annual Income (k$)", "mean"),
        Average_Spending=("Spending Score (1-100)", "mean")
    ).round(2)

    print("\n========== CLUSTER PROFILES ==========")

    print(profile)

    # --------------------------------------------------------
    # Gender distribution
    # --------------------------------------------------------

    gender_profile = pd.crosstab(
        result["Cluster"],
        result["Gender"],
        normalize="index"
    ).round(3)

    print("\n========== GENDER DISTRIBUTION ==========")

    print(gender_profile)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    profile_path = (
        RESULT_DIR
        / f"{configuration_name}_profile.csv"
    )

    gender_path = (
        RESULT_DIR
        / f"{configuration_name}_gender.csv"
    )

    customers_path = (
        RESULT_DIR
        / f"{configuration_name}_customers.csv"
    )

    profile.to_csv(profile_path)

    gender_profile.to_csv(gender_path)

    result.to_csv(
        customers_path,
        index=False
    )

    print("\n========== FILES SAVED ==========")

    print("Profile:", profile_path)

    print("Gender:", gender_path)

    print("Customers:", customers_path)

    print("Iterations:", model.n_iter_)

    print("Inertia:", round(model.inertia_, 4))

    print("\nConfiguration completed:", configuration_name)

    return profile, gender_profile, result


# ============================================================
# 6. MAIN
# ============================================================

def main():

    print("\n========================================")

    print("CUSTOMER CLUSTER PROFILE ANALYSIS")

    print("========================================")

    df = load_data()

    configurations = [
        (
            "without_gender_standard_k6",
            get_config("behavior_analysis")
        ),
        (
            "with_gender_minmax_k2",
            get_config("comparison_analysis")
        )
    ]

    for configuration_name, configuration in configurations:

        if not isinstance(configuration, dict):
            raise ValueError(
                f"Invalid configuration: {configuration_name}"
            )

        analyze_configuration(
            df,
            configuration_name,
            configuration
        )

    print("\n========================================")

    print("CLUSTER ANALYSIS COMPLETED!")

    print("========================================")


if __name__ == "__main__":
    main()
