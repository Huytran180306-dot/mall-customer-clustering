import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ==========================================
# 1. CONFIGURATION
# ==========================================

DATA_PATH = "data/raw/Mall_Customers.csv"

OUTPUT_DIR = "reports/figures/customer_segmentation/01_eda"

os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")


# ==========================================
# 2. LOAD DATA
# ==========================================

def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    print("\n========== DATASET OVERVIEW ==========")

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    return df


# ==========================================
# 3. DATA QUALITY ANALYSIS
# ==========================================

def analyze_data_quality(df):

    print("\n========== MISSING VALUES ==========")

    print(df.isnull().sum())

    print("\n========== DUPLICATES ==========")

    print(df.duplicated().sum())

    print("\n========== DATA TYPES ==========")

    print(df.dtypes)

    print("\n========== STATISTICS ==========")

    print(df.describe())


# ==========================================
# 4. GENDER DISTRIBUTION
# ==========================================

def plot_gender_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.countplot(
        data=df,
        x="Gender"
    )

    plt.title("Customer Gender Distribution")

    plt.xlabel("Gender")
    plt.ylabel("Number of Customers")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/gender_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 5. AGE DISTRIBUTION
# ==========================================

def plot_age_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["Age"],
        bins=15,
        kde=True
    )

    plt.title("Customer Age Distribution")

    plt.xlabel("Age")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/age_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 6. ANNUAL INCOME DISTRIBUTION
# ==========================================

def plot_income_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["Annual Income (k$)"],
        bins=15,
        kde=True
    )

    plt.title("Annual Income Distribution")

    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/income_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 7. SPENDING SCORE DISTRIBUTION
# ==========================================

def plot_spending_distribution(df):

    plt.figure(figsize=(8, 5))

    sns.histplot(
        df["Spending Score (1-100)"],
        bins=15,
        kde=True
    )

    plt.title("Spending Score Distribution")

    plt.xlabel("Spending Score")
    plt.ylabel("Frequency")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/spending_distribution.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 8. CORRELATION HEATMAP
# ==========================================

def plot_correlation_heatmap(df):

    numeric_df = df.select_dtypes(
        include="number"
    ).drop(
        columns=["CustomerID"],
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
        f"{OUTPUT_DIR}/correlation_heatmap.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 9. SCATTER PLOTS
# ==========================================

def plot_customer_scatter(df):

    plt.figure(figsize=(9, 6))

    sns.scatterplot(
        data=df,
        x="Annual Income (k$)",
        y="Spending Score (1-100)",
        hue="Gender",
        s=80
    )

    plt.title("Annual Income vs Spending Score")

    plt.tight_layout()

    plt.savefig(
        f"{OUTPUT_DIR}/income_spending_scatter.png",
        dpi=300
    )

    plt.close()


# ==========================================
# 10. BOXPLOTS - OUTLIER ANALYSIS
# ==========================================

def plot_outliers(df):

    features = [
        "Age",
        "Annual Income (k$)",
        "Spending Score (1-100)"
    ]

    for feature in features:

        plt.figure(figsize=(8, 5))

        sns.boxplot(
            x=df[feature]
        )

        plt.title(f"Outlier Analysis - {feature}")

        plt.tight_layout()

        filename = feature.replace(
            " ", "_"
        ).replace(
            "(", ""
        ).replace(
            ")", ""
        )

        plt.savefig(
            f"{OUTPUT_DIR}/{filename}_boxplot.png",
            dpi=300
        )

        plt.close()


# ==========================================
# 11. MAIN PIPELINE
# ==========================================

def main():

    print("\nSTARTING EDA PIPELINE...")

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

    print("================================")


if __name__ == "__main__":

    main()