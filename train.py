"""
train.py
========
End-to-end, reproducible training script for the Customer Churn
Prediction system.

Run:
    python train.py

Produces:
    models/customer_churn_pipeline.joblib   - complete sklearn Pipeline
                                               (preprocessing + trained model)
    models/model_metadata.json              - version info, features, metrics
    reports/model_evaluation.json           - full evaluation report
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.evaluation import evaluate_predictions, format_comparison_table
from src.preprocessing import (
    build_feature_lists,
    build_preprocessor,
    clean_data,
    encode_target,
    engineer_features,
    inspect_dataset,
    load_data,
)
from src.training import RANDOM_STATE, compare_models_cv, tune_model

DATA_PATH = Path("data/customer_churn.csv")
MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")
MODEL_PATH = MODEL_DIR / "customer_churn_pipeline.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
EVAL_REPORT_PATH = REPORT_DIR / "model_evaluation.json"


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    REPORT_DIR.mkdir(exist_ok=True)

    print("=" * 70)
    print("CUSTOMER CHURN PREDICTION — TRAINING PIPELINE")
    print("=" * 70)

    # ------------------------------------------------------------------
    # STEP 1: Load + Inspect
    # ------------------------------------------------------------------
    print("\n[1/9] Loading and inspecting dataset...")
    df_raw = load_data(str(DATA_PATH))
    profile = inspect_dataset(df_raw)

    print(f"  Shape: {profile.n_rows} rows x {profile.n_cols} columns")
    print(f"  Target column: '{profile.target_column}'")
    print(f"  Identifier column(s): {profile.id_columns}")
    print(f"  Numerical columns ({len(profile.numerical_columns)}): {profile.numerical_columns}")
    print(f"  Categorical columns ({len(profile.categorical_columns)}): {profile.categorical_columns}")
    print(f"  Duplicate rows: {profile.n_duplicates}")
    print(f"  Missing values by column: {profile.missing_summary}")
    print(f"  Target distribution: {profile.target_distribution}")
    for note in profile.notes:
        print(f"  NOTE: {note}")

    # ------------------------------------------------------------------
    # STEP 2: Clean
    # ------------------------------------------------------------------
    print("\n[2/9] Cleaning data...")
    df_clean, clean_log = clean_data(df_raw, profile)
    for entry in clean_log:
        print(f"  - {entry}")

    # ------------------------------------------------------------------
    # STEP 3: Target encoding
    # ------------------------------------------------------------------
    print("\n[3/9] Encoding target variable...")
    y, target_mapping = encode_target(df_clean, profile.target_column)
    print(f"  Target mapping applied: {target_mapping}")

    # ------------------------------------------------------------------
    # STEP 4: Feature engineering
    # ------------------------------------------------------------------
    print("\n[4/9] Engineering features...")
    df_engineered, eng_log = engineer_features(df_clean)
    for entry in eng_log:
        print(f"  - {entry}")

    numerical_cols, categorical_cols = build_feature_lists(
        profile, engineered_added=bool(eng_log)
    )

    drop_cols = profile.id_columns + [profile.target_column]
    feature_cols = [c for c in numerical_cols + categorical_cols if c in df_engineered.columns]
    X = df_engineered[feature_cols].copy()

    print(f"  Final feature set ({len(feature_cols)}): {feature_cols}")
    print(f"  Excluded as identifiers/leakage risk: {drop_cols}")

    # ------------------------------------------------------------------
    # STEP 5: Train/test split (BEFORE any preprocessing is fit)
    # ------------------------------------------------------------------
    print("\n[5/9] Splitting data (80/20, stratified, random_state=42)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

    # ------------------------------------------------------------------
    # STEP 6: Build preprocessing pipeline (fit only inside CV / on train)
    # ------------------------------------------------------------------
    print("\n[6/9] Building preprocessing pipeline...")
    preprocessor = build_preprocessor(numerical_cols, categorical_cols)
    print(f"  Numerical pipeline: median imputation + StandardScaler ({len(numerical_cols)} cols)")
    print(f"  Categorical pipeline: most-frequent imputation + OneHotEncoder ({len(categorical_cols)} cols)")

    # ------------------------------------------------------------------
    # STEP 7: Compare candidate models via stratified 5-fold CV
    # ------------------------------------------------------------------
    print("\n[7/9] Comparing candidate models with 5-fold stratified CV...")
    print("  (preprocessing is re-fit inside every fold — no leakage)")
    cv_results = compare_models_cv(preprocessor, X_train, y_train, n_splits=5)
    print()
    print(f"  {'Model':<22}{'ROC-AUC':>10}{'(+/-std)':>10}{'F1':>8}{'PR-AUC':>9}{'Acc':>8}")
    for r in cv_results:
        print(
            f"  {r.model_name:<22}{r.mean_roc_auc:>10.4f}{r.std_roc_auc:>10.4f}"
            f"{r.mean_f1:>8.4f}{r.mean_pr_auc:>9.4f}{r.mean_accuracy:>8.4f}"
        )

    best_candidate_name = cv_results[0].model_name
    print(f"\n  Selected for tuning (highest mean CV ROC-AUC, F1 tiebreaker): {best_candidate_name}")

    # ------------------------------------------------------------------
    # STEP 8: Hyperparameter tuning of the selected candidate
    # ------------------------------------------------------------------
    print(f"\n[8/9] Hyperparameter tuning '{best_candidate_name}' with GridSearchCV...")
    tuned_pipeline, best_params, best_cv_score = tune_model(
        preprocessor, best_candidate_name, X_train, y_train, n_splits=5
    )
    print(f"  Best params: {best_params}")
    print(f"  Best CV ROC-AUC: {best_cv_score}")

    # ------------------------------------------------------------------
    # STEP 9: Final evaluation on the held-out test set
    # ------------------------------------------------------------------
    print("\n[9/9] Evaluating final model on held-out test set...")
    y_pred = tuned_pipeline.predict(X_test)
    y_proba = None
    if hasattr(tuned_pipeline, "predict_proba"):
        y_proba = tuned_pipeline.predict_proba(X_test)[:, 1]

    final_eval = evaluate_predictions(best_candidate_name, y_test, y_pred, y_proba)
    print("\n  FINAL TEST-SET METRICS")
    print("  " + "-" * 50)
    for k, v in final_eval.to_dict().items():
        print(f"  {k}: {v}")

    # ------------------------------------------------------------------
    # Save artifacts
    # ------------------------------------------------------------------
    import joblib

    joblib.dump(tuned_pipeline, MODEL_PATH)
    print(f"\nSaved trained pipeline -> {MODEL_PATH}")

    # Allowed categorical values (from training data) for input validation
    categorical_allowed_values = {
        col: sorted(X_train[col].dropna().astype(str).unique().tolist())
        for col in categorical_cols
        if col in ("tenure_group",)  # derived field, always deterministic
    }
    raw_categorical_source_cols = [
        c for c in profile.categorical_columns if c in X_train.columns
    ]
    for col in raw_categorical_source_cols:
        categorical_allowed_values[col] = sorted(
            X_train[col].dropna().astype(str).unique().tolist()
        )

    metadata = {
        "model_name": f"CustomerChurnPipeline({best_candidate_name})",
        "base_estimator": best_candidate_name,
        "model_version": "1.0",
        "training_date": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "dataset": str(DATA_PATH),
        "dataset_source": "IBM Telco Customer Churn dataset (public)",
        "n_rows_total": profile.n_rows,
        "n_train_samples": int(X_train.shape[0]),
        "n_test_samples": int(X_test.shape[0]),
        "target": profile.target_column,
        "target_mapping": target_mapping,
        "feature_columns": feature_cols,
        "numerical_columns": numerical_cols,
        "categorical_columns": categorical_cols,
        "categorical_allowed_values": categorical_allowed_values,
        "identifier_columns_excluded": profile.id_columns,
        "cv_model_comparison": [r.__dict__ for r in cv_results],
        "best_params": best_params,
        "best_cv_roc_auc": best_cv_score,
        "test_metrics": final_eval.to_dict(),
        "random_state": RANDOM_STATE,
    }

    # raw_input_fields should exclude engineered features (the UI collects
    # raw fields only; engineered features are derived automatically).
    engineered_names = {"avg_monthly_charge", "tenure_group", "total_services"}
    metadata["raw_input_fields"] = [c for c in feature_cols if c not in engineered_names]
    metadata["engineered_feature_names"] = [c for c in feature_cols if c in engineered_names]

    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"Saved model metadata -> {METADATA_PATH}")

    full_report = {
        "cv_results": [r.__dict__ for r in cv_results],
        "final_model": final_eval.to_dict(),
        "best_params": best_params,
        "cleaning_log": clean_log,
        "feature_engineering_log": eng_log,
        "dataset_profile": {
            "n_rows": profile.n_rows,
            "n_cols": profile.n_cols,
            "n_duplicates": profile.n_duplicates,
            "missing_summary": profile.missing_summary,
            "target_distribution": profile.target_distribution,
            "notes": profile.notes,
        },
    }
    with open(EVAL_REPORT_PATH, "w") as f:
        json.dump(full_report, f, indent=2, default=str)
    print(f"Saved evaluation report -> {EVAL_REPORT_PATH}")

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Final model: {best_candidate_name}")
    print(f"Test ROC-AUC: {final_eval.roc_auc:.4f} | Test F1: {final_eval.f1:.4f} | "
          f"Test Accuracy: {final_eval.accuracy:.4f}")
    print(f"Artifact ready at: {MODEL_PATH}")
    print("Run `streamlit run app.py` to launch the application.")


if __name__ == "__main__":
    main()
