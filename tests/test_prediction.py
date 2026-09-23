"""
tests/test_prediction.py
=========================
Automated tests for the Customer Churn Prediction system.

Run:
    python -m pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prediction import (
    MODEL_PATH,
    ValidationError,
    load_model_and_metadata,
    predict_customer,
)

VALID_CUSTOMER = {
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
}


@pytest.fixture(scope="module")
def model_and_metadata():
    model, metadata = load_model_and_metadata()
    return model, metadata


# ---------------------------------------------------------------------
# Test 1: Model artifact exists
# ---------------------------------------------------------------------
def test_model_artifact_exists():
    assert MODEL_PATH.exists(), (
        f"Model artifact not found at {MODEL_PATH}. Run `python train.py` first."
    )


# ---------------------------------------------------------------------
# Test 2: Model can be loaded
# ---------------------------------------------------------------------
def test_model_can_be_loaded(model_and_metadata):
    model, metadata = model_and_metadata
    assert model is not None
    assert hasattr(model, "predict")
    assert "model_name" in metadata


# ---------------------------------------------------------------------
# Test 3: A valid customer record produces a prediction
# ---------------------------------------------------------------------
def test_valid_customer_produces_prediction(model_and_metadata):
    model, metadata = model_and_metadata
    result = predict_customer(VALID_CUSTOMER, model, metadata)
    assert "prediction" in result
    assert result["prediction"] is not None


# ---------------------------------------------------------------------
# Test 4: Prediction is one of the expected classes
# ---------------------------------------------------------------------
def test_prediction_is_valid_class(model_and_metadata):
    model, metadata = model_and_metadata
    result = predict_customer(VALID_CUSTOMER, model, metadata)
    assert result["prediction"] in (0, 1)
    assert result["prediction_label"] in ("Likely to Churn", "Not Likely to Churn")


# ---------------------------------------------------------------------
# Test 5: Probability values are valid if probability prediction is supported
# ---------------------------------------------------------------------
def test_probabilities_are_valid(model_and_metadata):
    model, metadata = model_and_metadata
    result = predict_customer(VALID_CUSTOMER, model, metadata)
    if result["probabilities"] is not None:
        probs = list(result["probabilities"].values())
        assert all(0.0 <= p <= 1.0 for p in probs)
        assert abs(sum(probs) - 1.0) < 1e-6
        assert 0.0 <= result["churn_probability"] <= 1.0


# ---------------------------------------------------------------------
# Test 6: Invalid user input is handled correctly (no crash, clear error)
# ---------------------------------------------------------------------
def test_invalid_numeric_input_raises_validation_error(model_and_metadata):
    model, metadata = model_and_metadata
    bad_customer = dict(VALID_CUSTOMER)
    bad_customer["tenure"] = -5  # invalid: negative tenure
    with pytest.raises(ValidationError):
        predict_customer(bad_customer, model, metadata)


def test_non_numeric_value_in_numeric_field_raises_validation_error(model_and_metadata):
    model, metadata = model_and_metadata
    bad_customer = dict(VALID_CUSTOMER)
    bad_customer["MonthlyCharges"] = "not-a-number"
    with pytest.raises(ValidationError):
        predict_customer(bad_customer, model, metadata)


def test_unexpected_categorical_value_raises_validation_error(model_and_metadata):
    model, metadata = model_and_metadata
    bad_customer = dict(VALID_CUSTOMER)
    bad_customer["Contract"] = "Lifetime"  # not a value seen during training
    with pytest.raises(ValidationError):
        predict_customer(bad_customer, model, metadata)


# ---------------------------------------------------------------------
# Test 7: The prediction pipeline does not require retraining
# ---------------------------------------------------------------------
def test_prediction_does_not_retrain(model_and_metadata):
    """
    Calling predict_customer repeatedly must not mutate or refit the
    loaded model — predictions for the same input must be identical
    across repeated calls (deterministic inference, no retraining).
    """
    model, metadata = model_and_metadata
    result_1 = predict_customer(VALID_CUSTOMER, model, metadata)
    result_2 = predict_customer(VALID_CUSTOMER, model, metadata)
    assert result_1["prediction"] == result_2["prediction"]
    if result_1["probabilities"] and result_2["probabilities"]:
        for k in result_1["probabilities"]:
            assert abs(result_1["probabilities"][k] - result_2["probabilities"][k]) < 1e-9


# ---------------------------------------------------------------------
# Extra: consistency between two independently loaded model instances
# (simulates predict.py vs. app.py both loading the same artifact)
# ---------------------------------------------------------------------
def test_two_independent_loads_agree():
    model_a, meta_a = load_model_and_metadata()
    model_b, meta_b = load_model_and_metadata()

    result_a = predict_customer(VALID_CUSTOMER, model_a, meta_a)
    result_b = predict_customer(VALID_CUSTOMER, model_b, meta_b)

    assert result_a["prediction"] == result_b["prediction"]
    if result_a["probabilities"] and result_b["probabilities"]:
        for k in result_a["probabilities"]:
            assert abs(result_a["probabilities"][k] - result_b["probabilities"][k]) < 1e-9
