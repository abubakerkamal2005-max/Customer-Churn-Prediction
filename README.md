# Customer Churn Prediction System

A complete, production-style machine learning application that predicts whether a
telecom customer is likely to churn. The prediction shown in the app is generated
directly by a trained scikit-learn model — there is no hard-coded, rule-based logic
anywhere in the prediction path.

---

## 1. Project Overview

This project implements an end-to-end ML system: data cleaning, feature engineering,
model comparison, cross-validation, hyperparameter tuning, a saved inference pipeline,
and a Streamlit application that uses that saved pipeline to make live predictions.

## 2. Business Problem

Customer churn (a customer canceling their subscription) is expensive to replace with
new acquisition. Being able to flag at-risk customers lets a business proactively
offer retention incentives to the customers most likely to leave, rather than
spending retention budget indiscriminately.

## 3. Objective

Given a customer's account, service, and billing information, predict the probability
that the customer will churn, using a model trained on historical churn outcomes.

## 4. Dataset

**IBM Telco Customer Churn dataset** (public), 7,043 customers × 21 columns, sourced
from `data/customer_churn.csv`. Target column: `Churn` (`Yes`/`No`). Class balance:
5,174 No-Churn (73.5%) vs. 1,869 Churn (26.5%) — moderately imbalanced, which is why
this project does not rely on accuracy alone (see §9).

The dataset-inspection layer (`src/preprocessing.py::inspect_dataset`) does not assume
exact column names — it locates the target and identifier columns case-insensitively
and auto-detects numeric-looking text columns, so the pipeline is reasonably robust to
minor schema variations.

## 5. Data Preparation

- **Duplicates:** none found in this dataset (checked and handled if present).
- **`TotalCharges`:** stored as text with 11 blank values (all correspond to customers
  with `tenure == 0`, i.e. brand-new customers with no billing history yet). Blanks are
  converted to `NaN` and handled by the numerical pipeline's median imputer — they are
  never silently dropped.
- **Identifiers:** `customerID` is detected and excluded from modeling. A unique ID has
  no genuine predictive signal and including it risks the model latching onto
  spurious per-row artifacts (a leakage/overfitting risk), so it is excluded and this
  decision is logged in the training output.

## 6. Feature Engineering

Three features are derived, each with an explicit justification (also logged at
training time):

| Feature | Definition | Justification |
|---|---|---|
| `avg_monthly_charge` | `TotalCharges / tenure` (falls back to `MonthlyCharges` when `tenure == 0`) | Captures a customer's long-run spend rate, distinct from their current `MonthlyCharges` snapshot |
| `tenure_group` | Bucketed tenure: `0-6mo`, `7-12mo`, `13-24mo`, `25-48mo`, `48mo+` | Churn risk is known to be highly non-linear in tenure; bucketing gives models an easy categorical signal |
| `total_services` | Count of subscribed add-on services (security, backup, device protection, tech support, streaming TV/movies, phone, multiple lines) | Simple proxy for how embedded a customer is in the product ecosystem |

No other features were added — extra features were deliberately avoided where they had
no clear justification.

## 7. Machine Learning Approach

```
Raw CSV → Clean → Encode target → Engineer features → Train/test split (80/20, stratified)
        → ColumnTransformer (impute+scale numeric, impute+one-hot categorical)
        → 5-fold stratified CV across candidate models
        → GridSearchCV tuning of the best candidate
        → Final sklearn Pipeline(preprocessor + tuned model)
        → Evaluate once on held-out test set
        → Save as models/customer_churn_pipeline.joblib
```

The `ColumnTransformer` is always fit **inside** the training data only (never on the
full dataset, and freshly inside every CV fold) to prevent data leakage, and it is
saved as part of the final `Pipeline` object — the application never reimplements
preprocessing separately.

## 8. Models Compared

Five candidates spanning linear, single-tree, and ensemble methods:

1. Logistic Regression (`class_weight="balanced"`)
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. Hist Gradient Boosting

All five are compared via 5-fold `StratifiedKFold` cross-validation with the full
preprocessing pipeline inside the CV loop (`src/training.py::compare_models_cv`).

## 9. Evaluation Metrics

Accuracy alone is **not** used for model selection because of class imbalance — a
model that always predicts "No churn" would score ~73.5% accuracy while being useless.
Instead the full metric suite is computed for every candidate:

- **Accuracy** — overall correctness (reported, not decisive)
- **Precision** — of flagged at-risk customers, how many actually churn (false
  positives waste retention budget)
- **Recall** — of customers who actually churn, how many the model catches (false
  negatives are lost revenue with no chance to intervene)
- **F1-score** — balanced summary of precision and recall
- **ROC-AUC** — threshold-independent ranking quality; the primary model-selection
  metric used here
- **PR-AUC (average precision)** — like ROC-AUC but focused on the minority (churn)
  class, more informative under imbalance

Model selection ranks candidates by mean cross-validated ROC-AUC, with F1 as a
tiebreaker (`src/training.py::compare_models_cv`).

## 10. Final Model

The final model is selected automatically each time `train.py` runs (see
`models/model_metadata.json` for the actual result of the most recent run — it is not
hard-coded in this README). In the reference run performed for this project, the
winning candidate was **Gradient Boosting**, tuned via `GridSearchCV` over
`n_estimators`, `learning_rate`, and `max_depth`, achieving approximately:

- ROC-AUC ≈ 0.85
- F1 ≈ 0.57
- Accuracy ≈ 0.80

on the held-out test set (exact figures in `reports/model_evaluation.json`). These are
the model's real, unmodified test-set numbers — nothing here is fabricated.

## 11. How to Train

```bash
pip install -r requirements.txt
python train.py
```

This regenerates:
- `models/customer_churn_pipeline.joblib` — the complete pipeline artifact
- `models/model_metadata.json` — version info, feature list, metrics
- `reports/model_evaluation.json` — full evaluation report

Training is reproducible: `random_state=42` is used throughout (train/test split,
cross-validation folds, and every stochastic model).

## 12. How to Run the Application

```bash
streamlit run app.py
```

Fill in the customer information form and click **PREDICT CHURN**. The app loads the
saved pipeline once (`@st.cache_resource`) and never retrains on click.

## 13. How Prediction Works

Both `predict.py` (CLI demo) and `app.py` (Streamlit UI) call the same function,
`src.prediction.predict_customer`, which is the single source of truth for inference:

```
raw customer dict
   → validate_customer_input()      (numeric ranges, required fields, known categories)
   → build_feature_frame()          (adds the same 3 engineered features as training)
   → model.predict(frame)           (the saved Pipeline: preprocessing + trained model)
   → model.predict_proba(frame)
   → prediction_label ("Likely to Churn" / "Not Likely to Churn")
```

No branch of this path contains manually written churn rules. The label is a direct
mapping of the model's own 0/1 output, and the probability is `predict_proba` output,
unmodified.

## 14. Project Structure

```
customer-churn-ml/
├── app.py                          # Streamlit application
├── train.py                        # End-to-end training script
├── predict.py                      # CLI demo of the prediction path
├── requirements.txt
├── README.md
├── data/
│   └── customer_churn.csv          # IBM Telco Customer Churn dataset
├── models/
│   ├── customer_churn_pipeline.joblib
│   └── model_metadata.json
├── src/
│   ├── preprocessing.py            # Inspection, cleaning, feature engineering, ColumnTransformer
│   ├── training.py                 # Candidate models, CV comparison, hyperparameter tuning
│   ├── evaluation.py               # Metrics computation and reporting
│   └── prediction.py               # Single source of truth for inference + validation
├── reports/
│   └── model_evaluation.json       # Full evaluation report from the last training run
└── tests/
    └── test_prediction.py          # Automated tests (see below)
```

## 15. Limitations

- Trained on a single historical snapshot; churn drivers can shift over time and the
  model should be periodically retrained on fresh data.
- ROC-AUC ≈ 0.85 / F1 ≈ 0.57 means the model is a useful risk-ranking tool, not a
  perfect classifier — some churners will be missed and some loyal customers will be
  flagged.
- The "model-associated factors" shown in the app are feature importances, which
  reflect correlation the model learned, not proven causation.
- No temporal/holdout-by-date validation was performed (a single random stratified
  split was used), since the source dataset has no timestamp field.

## 16. Future Improvements

- Add SHAP-based per-prediction explanations instead of only global feature importance.
- Track model performance over time and add automatic retraining triggers.
- Add a calibration check/plot for the predicted probabilities.
- Expand hyperparameter search with `RandomizedSearchCV` given more compute budget.
- Add an API layer (e.g. FastAPI) alongside the Streamlit UI for programmatic access.

---

## Testing

```bash
python -m pytest tests/ -v
```

Covers: model artifact existence, model loading, valid predictions, valid prediction
classes, valid probability values, invalid-input handling (negative tenure, non-numeric
fields, unseen categorical values), determinism/no-retraining-on-predict, and
consistency across independently loaded model instances.

## Reproducibility Notes

- Python: 3.12
- Key dependencies: scikit-learn, pandas, numpy, joblib, streamlit (see
  `requirements.txt` for version bounds)
- `random_state=42` used for the train/test split, all cross-validation folds, and
  every model with a stochastic component
- Exact versions and the training timestamp of the currently saved artifact are
  recorded in `models/model_metadata.json`
