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