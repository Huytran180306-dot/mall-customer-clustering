import sys
from pathlib import Path

import joblib
import pandas as pd


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT_DIR = Path(__file__).resolve().parents[1]

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Register custom KMeans class for loading the saved model
from src.models.kmeans import KMeans


MODEL_DIR = ROOT_DIR / "models/kmeans"

MODEL_PATH = MODEL_DIR / "mall_customer_kmeans.pkl"

SCALER_PATH = MODEL_DIR / "minmax_scaler.pkl"


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
# 2. LOAD MODEL
# ==========================================

def load_model():

    print("\nLOADING TRAINED MODEL...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)

    if not SCALER_PATH.exists():
        raise FileNotFoundError(SCALER_PATH)

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    print("MODEL LOADED SUCCESSFULLY!")

    return model, scaler


# ==========================================
# 3. GET CUSTOMER INFORMATION
# ==========================================

def get_customer_input():

    print("\nENTER CUSTOMER INFORMATION")

    while True:

        gender = input(
            "Gender (Male/Female): "
        ).strip().title()

        if gender in GENDER_MAPPING:
            break

        print("Invalid gender. Please try again.")

    age = float(input("Age: "))

    income = float(
        input("Annual Income (k$): ")
    )

    spending = float(
        input("Spending Score (1-100): ")
    )

    if age <= 0 or income <= 0:
        raise ValueError("Age and income must be positive.")

    if not 1 <= spending <= 100:
        raise ValueError("Spending Score must be between 1 and 100.")

    customer = pd.DataFrame(
        [[
            GENDER_MAPPING[gender],
            age,
            income,
            spending
        ]],
        columns=FEATURE_COLUMNS
    )

    return customer


# ==========================================
# 4. PREDICT CUSTOMER CLUSTER
# ==========================================

def predict_customer(model, scaler, customer):

    print("\nPROCESSING CUSTOMER DATA...")

    scaled_customer = scaler.transform(customer)

    cluster = model.predict(scaled_customer)

    return int(cluster[0])


# ==========================================
# 5. MAIN
# ==========================================

def main():

    print("=" * 45)

    print("MALL CUSTOMER CLUSTER PREDICTION")

    print("=" * 45)

    model, scaler = load_model()

    customer = get_customer_input()

    cluster = predict_customer(
        model,
        scaler,
        customer
    )

    print("\n" + "=" * 45)

    print("PREDICTION RESULT")

    print("=" * 45)

    print("\nCustomer Information:")

    print(customer.to_string(index=False))

    print("\nPredicted Cluster:", cluster)

    print("\nPREDICTION COMPLETED SUCCESSFULLY!")


if __name__ == "__main__":

    main()