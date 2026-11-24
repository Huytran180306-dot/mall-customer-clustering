"""
Exploratory Data Analysis module for Mall Customer Clustering.

Responsibilities:
1. Load customer data.
2. Analyze data quality and statistical characteristics.
3. Visualize demographic distributions.
4. Analyze feature correlations.
5. Explore customer income and spending relationships.
6. Detect potential outliers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================
# 1. CENTRALIZED CONFIGURATION
# ============================================================

ROOT_DIR = Path(__file__).resolve().parent.parent

from src.config import get_config, get_path


DATA_PATH = get_path("data", "raw_path")

OUTPUT_DIR = (
    get_path("paths", "figures")
    / "01_eda"
)

CUSTOMER_ID_COLUMN = get_config(
    "data", "id_column"
)

CATEGORICAL_FEATURES = get_config(
    "data", "categorical_features"
)

NUMERIC_FEATURES = get_config(
    "data", "numeric_features"
)

GENDER_COLUMN = CATEGORICAL_FEATURES[0]

AGE_COLUMN = NUMERIC_FEATURES[0]

INCOME_COLUMN = NUMERIC_FEATURES[1]

SPENDING_COLUMN = NUMERIC_FEATURES[2]

# Outlier analysis configuration
RESULT_DIR = get_path("paths", "progress")

OUTLIER_CONFIG = get_config(
    "outlier_analysis",
    default={}
)

OUTLIER_ENABLED = OUTLIER_CONFIG.get(
    "enabled",
    True
)

OUTLIER_METHOD = OUTLIER_CONFIG.get(
    "method",
    "IQR"
)

IQR_MULTIPLIER = float(
    OUTLIER_CONFIG.get("multiplier", 1.5)
)

OUTLIER_FEATURES = get_config(
    "outlier_analysis",
    "features",
    default=NUMERIC_FEATURES
)

OUTLIER_SUMMARY_PATH = (
    RESULT_DIR / "outlier_analysis.csv"
)

OUTLIER_RECORDS_PATH = (
    RESULT_DIR / "outlier_records.csv"
)


# ============================================================
# 2. PREPARE DIRECTORIES AND VISUALIZATION SETTINGS
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

sns.set_theme(style="whitegrid")


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data():

    print("\n========== DATASET OVERVIEW ==========")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = [
        CUSTOMER_ID_COLUMN,
        GENDER_COLUMN,
        AGE_COLUMN,
        INCOME_COLUMN,
        SPENDING_COLUMN
    ]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ============================================================
# 4. DATA QUALITY ANALYSIS
# ============================================================

def analyze_data_quality(df):


    print("\n========== MISSING VALUES ==========")

    print(df.isnull().sum())

    print("\n========== DUPLICATES ==========")

    print(df.duplicated().sum())

    print("\n========== DATA TYPES ==========")

    print(df.dtypes)

    print("\n========== STATISTICS ==========")

    print(df.describe())

# ============================================================
# OUTLIER DETECTION - IQR METHOD
# ============================================================

def analyze_outliers_iqr(df):

    print("\n========== OUTLIER ANALYSIS - IQR ==========")

    if not OUTLIER_ENABLED:
        print("Outlier analysis is disabled.")
        return None

    if OUTLIER_METHOD.upper() != "IQR":
        raise ValueError(
            f"Unsupported method: {OUTLIER_METHOD}"
        )

    if IQR_MULTIPLIER <= 0:
        raise ValueError(
            "IQR multiplier must be greater than 0."
        )

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    summary_records = []
    detail_records = []

    for feature in OUTLIER_FEATURES:

        if feature not in df.columns:
            raise ValueError(
                f"Feature not found: {feature}"
            )

        values = pd.to_numeric(
            df[feature],
            errors="coerce"
        )

        invalid_mask = (
            df[feature].notna()
            & values.isna()
        )

        if invalid_mask.any():
            raise ValueError(
                f"Non-numeric values in {feature}"
            )

        valid_values = values.dropna()

        if valid_values.empty:
            raise ValueError(
                f"No valid values in {feature}"
            )

        if not np.isfinite(valid_values.to_numpy()).all():
            raise ValueError(
                f"Infinite values detected in {feature}"
            )

        # Calculate quartiles
        q1 = valid_values.quantile(0.25)
        q3 = valid_values.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - IQR_MULTIPLIER * iqr
        upper_bound = q3 + IQR_MULTIPLIER * iqr

        # Detect outliers
        outlier_mask = (
            (values < lower_bound)
            | (values > upper_bound)
        ).fillna(False)

        outlier_count = int(outlier_mask.sum())
        valid_count = len(valid_values)

        percentage = (
            outlier_count / valid_count * 100
        )

        # Save summary
        summary_records.append({

            "Feature": feature,
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower_Bound": lower_bound,
            "Upper_Bound": upper_bound,
            "Minimum": valid_values.min(),
            "Maximum": valid_values.max(),
            "Valid_Count": valid_count,
            "Outlier_Count": outlier_count,
            "Outlier_Percentage": percentage

        })

        # Save individual outliers
        for row_pos in range(len(df)):

            if not outlier_mask.iloc[row_pos]:
                continue

            value = values.iloc[row_pos]

            detail_records.append({

                "Row_Index": row_pos,

                "CustomerID": df.iloc[row_pos][
                    CUSTOMER_ID_COLUMN
                ],

                "Feature": feature,

                "Value": value,

                "Lower_Bound": lower_bound,

                "Upper_Bound": upper_bound,

                "Outlier_Type": (
                    "Below Lower Bound"
                    if value < lower_bound
                    else "Above Upper Bound"
                )

            })

        print(f"\nFeature: {feature}")
        print(f"Q1: {q1:.2f}")
        print(f"Q3: {q3:.2f}")
        print(f"IQR: {iqr:.2f}")
        print(f"Lower Bound: {lower_bound:.2f}")
        print(f"Upper Bound: {upper_bound:.2f}")
        print(f"Outliers: {outlier_count}")
        print(f"Percentage: {percentage:.2f}%")

    # Export results
    summary_df = pd.DataFrame(summary_records)

    details_df = pd.DataFrame(
        detail_records,
        columns=[
            "Row_Index",
            "CustomerID",
            "Feature",
            "Value",
            "Lower_Bound",
            "Upper_Bound",
            "Outlier_Type"
        ]
    )

    summary_df.to_csv(
        OUTLIER_SUMMARY_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    details_df.to_csv(
        OUTLIER_RECORDS_PATH,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n========== OUTLIER SUMMARY ==========")

    print(
        summary_df[
            [
                "Feature",
                "Outlier_Count",
                "Outlier_Percentage"
            ]
        ].to_string(index=False)
    )

    print("\nSummary saved:", OUTLIER_SUMMARY_PATH)
    print("Records saved:", OUTLIER_RECORDS_PATH)

    return summary_df, details_df


# ============================================================
# 5. GENDER DISTRIBUTION
# ============================================================

def plot_gender_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x=GENDER_COLUMN
    )

    plt.title("Customer Gender Distribution")

    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "gender_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 6. AGE DISTRIBUTION
# ============================================================

def plot_age_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[AGE_COLUMN],
        bins=15,
        kde=True
    )

    plt.title("Customer Age Distribution")

    plt.xlabel("Age")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "age_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 7. ANNUAL INCOME DISTRIBUTION
# ============================================================

def plot_income_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[INCOME_COLUMN],
        bins=15,
        kde=True
    )

    plt.title("Annual Income Distribution")

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "income_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 8. SPENDING SCORE DISTRIBUTION
# ============================================================

def plot_spending_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df[SPENDING_COLUMN],
        bins=15,
        kde=True
    )

    plt.title("Spending Score Distribution")

    plt.xlabel("Spending Score")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "spending_distribution.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 9. CORRELATION HEATMAP
# ============================================================

def plot_correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include="number"
    ).drop(
        columns=[CUSTOMER_ID_COLUMN],
        errors="ignore"
    )

    plt.figure(figsize=(9, 6))

    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        fmt=".2f"
    )

    plt.title("Feature Correlation Heatmap")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 10. CUSTOMER SCATTER PLOT
# ============================================================

def plot_customer_scatter(df):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x=INCOME_COLUMN,
        y=SPENDING_COLUMN,
        hue=GENDER_COLUMN,
        s=80
    )

    plt.title("Annual Income vs Spending Score")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "income_spending_scatter.png",
        dpi=300
    )

    plt.close()


# ============================================================
# 11. BOXPLOTS - OUTLIER ANALYSIS
# ============================================================

def plot_outliers(df):

    features = [
        AGE_COLUMN,
        INCOME_COLUMN,
        SPENDING_COLUMN
    ]

    for feature in features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            x=df[feature]
        )

        plt.title(f"Outlier Analysis - {feature}")

        plt.tight_layout()

        filename = (
            feature.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
        )

        plt.savefig(
            OUTPUT_DIR / f"{filename}_boxplot.png",
            dpi=300
        )

        plt.close()


# ============================================================
# 12. MAIN PIPELINE
# ============================================================

def main():

    print("\nSTARTING EDA PIPELINE...")

    print("Dataset:", DATA_PATH)

    print("Output directory:", OUTPUT_DIR)

    df = load_data()

    analyze_data_quality(df)

    analyze_outliers_iqr(df)

    plot_gender_distribution(df)

    plot_age_distribution(df)

    plot_income_distribution(df)

    plot_spending_distribution(df)

    plot_correlation_heatmap(df)

    plot_customer_scatter(df)

    plot_outliers(df)

    print("\n================================")

    print("EDA COMPLETED SUCCESSFULLY!")

    print("Figures saved at:", OUTPUT_DIR)

    print("Total figures generated: 9")

    print("================================")


if __name__ == "__main__":

    main()