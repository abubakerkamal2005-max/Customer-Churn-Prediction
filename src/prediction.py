"""
src/prediction.py
==================
Single source of truth for turning raw customer input into a model
prediction. Both predict.py (CLI demo) and app.py (Streamlit UI) call
into this module so there is exactly one prediction code path — no
separate rule-based logic anywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.preprocessing import engineer_features

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "customer_churn_pipeline.joblib"
METADATA_PATH = Path(__file__).resolve().parent.parent / "models" / "model_metadata.json"


class ValidationError(Exception):
    """Raised when raw user input fails validation before being sent to the model."""


def load_model_and_metadata(
    model_path: Path | str = MODEL_PATH, metadata_path: Path | str = METADATA_PATH
):
    """Load the trained pipeline artifact and its accompanying metadata."""
    model_path = Path(model_path)
    metadata_path = Path(metadata_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model artifact not found at {model_path}. Run `python train.py` first."
        )

    model = joblib.load(model_path)

    metadata = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    return model, metadata


def validate_customer_input(raw: dict, metadata: dict) -> dict:
    """
    Validate raw user input (e.g. from the Streamlit form) against basic
    business rules and, where available, the categorical values the model
    was trained on. Raises ValidationError with a friendly message on
    failure. Returns a cleaned copy of the input dict.
    """
    cleaned = dict(raw)
    errors: list[str] = []

    numeric_fields = {
        "tenure": (0, 130),
        "MonthlyCharges": (0, 10000),
        "TotalCharges": (0, 1_000_000),
    }

    for field, (lo, hi) in numeric_fields.items():
        if field not in cleaned:
            continue
        value = cleaned[field]
        try:
            value = float(value)
        except (TypeError, ValueError):
            errors.append(f"'{field}' must be a valid number.")
            continue
        if value < lo or value > hi:
            errors.append(f"'{field}' must be between {lo} and {hi} (got {value}).")
        cleaned[field] = value

    if "TotalCharges" in cleaned and "MonthlyCharges" in cleaned and "tenure" in cleaned:
        if cleaned["tenure"] > 0 and cleaned["TotalCharges"] < 0:
            errors.append("'TotalCharges' cannot be negative.")

    allowed_values = metadata.get("categorical_allowed_values", {})
    for field, allowed in allowed_values.items():
        if field in cleaned and cleaned[field] not in allowed:
            errors.append(
                f"'{field}' has an unexpected value '{cleaned[field]}'. "
                f"Expected one of: {allowed}."
            )

    required_fields = metadata.get("raw_input_fields", [])
    for field in required_fields:
        if field not in cleaned or cleaned[field] in (None, ""):
            errors.append(f"'{field}' is required.")

    if errors:
        raise ValidationError(" | ".join(errors))

    return cleaned


def build_feature_frame(raw: dict) -> pd.DataFrame:
    """
    Turn a single raw customer record (dict) into the exact DataFrame
    structure the saved pipeline expects: one row, with the same
    engineered features created identically to training time.
    """
    df = pd.DataFrame([raw])
    df, _log = engineer_features(df)
    return df


def predict_customer(raw: dict, model, metadata: dict | None = None) -> dict:
    """
    Run the full, single-source-of-truth prediction path:
        raw dict -> validated -> feature frame -> model.predict / predict_proba

    Returns a dict with the raw model outputs plus a human-readable label.
    No business rules are applied here; the label is a direct mapping of
    the model's own 0/1 output.
    """
    metadata = metadata or {}
    validated = validate_customer_input(raw, metadata)
    frame = build_feature_frame(validated)

    prediction = model.predict(frame)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        proba_arr = model.predict_proba(frame)[0]
        # class order follows model.classes_
        classes = list(model.classes_)
        proba = {str(c): float(p) for c, p in zip(classes, proba_arr)}

    label = "Likely to Churn" if int(prediction) == 1 else "Not Likely to Churn"

    result = {
        "prediction": int(prediction),
        "prediction_label": label,
        "probabilities": proba,
    }

    if proba is not None:
        churn_key = "1" if "1" in proba else [k for k in proba if k.lower() == "yes"][0]
        result["churn_probability"] = proba.get(churn_key)

    return result
