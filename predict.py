"""
predict.py
==========
Demonstrates loading the trained pipeline artifact and making a
prediction from a raw customer record — using exactly the same input
structure and prediction path (src.prediction.predict_customer) as the
Streamlit application in app.py.

Run:
    python predict.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.prediction import load_model_and_metadata, predict_customer

# Example raw customer records (same field names the Streamlit app collects).
EXAMPLE_CUSTOMERS = [
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 1,
        "PhoneService": "No",
        "MultipleLines": "No phone service",
        "InternetService": "DSL",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 29.85,
        "TotalCharges": 29.85,
    },
    {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 34,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "No",
        "DeviceProtection": "Yes",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "One year",
        "PaperlessBilling": "No",
        "PaymentMethod": "Mailed check",
        "MonthlyCharges": 56.95,
        "TotalCharges": 1889.50,
    },
    {
        "gender": "Female",
        "SeniorCitizen": 1,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 72,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Credit card (automatic)",
        "MonthlyCharges": 105.65,
        "TotalCharges": 7600.0,
    },
]


def main() -> None:
    model, metadata = load_model_and_metadata()
    print(f"Loaded model: {metadata.get('model_name', 'unknown')}")
    print(f"Trained: {metadata.get('training_date', 'unknown')}")
    print("-" * 70)

    for i, customer in enumerate(EXAMPLE_CUSTOMERS, start=1):
        result = predict_customer(customer, model, metadata)
        print(f"Customer #{i}")
        print(f"  Prediction: {result['prediction']} -> {result['prediction_label']}")
        if result["probabilities"]:
            print(f"  Probabilities: {result['probabilities']}")
        print("-" * 70)


if __name__ == "__main__":
    main()
