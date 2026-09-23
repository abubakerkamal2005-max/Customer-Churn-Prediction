"""
src/training.py
================
Candidate model definitions, cross-validated model comparison, and
hyperparameter tuning for the selected candidate. The full preprocessing
pipeline is always included inside the cross-validated estimator, so
there is no leakage between folds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42

CV_SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "average_precision": "average_precision",
}


def get_candidate_models() -> dict:
    """
    Return the candidate classification algorithms compared during model
    selection. Chosen to span a representative range of approaches:
    a linear baseline, a single tree (for interpretability comparison),
    and three tree-ensemble methods of increasing sophistication.
    """
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
    }


@dataclass
class CVComparisonRow:
    model_name: str
    mean_roc_auc: float
    std_roc_auc: float
    mean_f1: float
    std_f1: float
    mean_pr_auc: float
    mean_accuracy: float
    mean_precision: float
    mean_recall: float


def compare_models_cv(
    preprocessor, X_train, y_train, n_splits: int = 5
) -> list[CVComparisonRow]:
    """
    Run stratified k-fold cross-validation for every candidate model, with
    preprocessing fit fresh inside every fold (no leakage). Models are
    ranked by mean ROC-AUC, a threshold-independent metric appropriate for
    an imbalanced binary classification problem, with F1 as a tiebreaker.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    rows: list[CVComparisonRow] = []

    for name, model in get_candidate_models().items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=skf, scoring=CV_SCORING, n_jobs=-1
        )
        rows.append(
            CVComparisonRow(
                model_name=name,
                mean_roc_auc=float(np.mean(scores["test_roc_auc"])),
                std_roc_auc=float(np.std(scores["test_roc_auc"])),
                mean_f1=float(np.mean(scores["test_f1"])),
                std_f1=float(np.std(scores["test_f1"])),
                mean_pr_auc=float(np.mean(scores["test_average_precision"])),
                mean_accuracy=float(np.mean(scores["test_accuracy"])),
                mean_precision=float(np.mean(scores["test_precision"])),
                mean_recall=float(np.mean(scores["test_recall"])),
            )
        )

    rows.sort(key=lambda r: (r.mean_roc_auc, r.mean_f1), reverse=True)
    return rows


def get_param_grid(model_name: str) -> dict:
    """
    Small, deliberately bounded hyperparameter grids for the top
    candidates. Kept modest to avoid an unnecessarily large search space.
    """
    grids = {
        "RandomForest": {
            "model__n_estimators": [200, 400],
            "model__max_depth": [None, 8, 16],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "GradientBoosting": {
            "model__n_estimators": [100, 200],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
        "HistGradientBoosting": {
            "model__max_iter": [150, 300],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [None, 6],
        },
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "DecisionTree": {
            "model__max_depth": [4, 6, 8],
            "model__min_samples_leaf": [1, 5, 10],
        },
    }
    return grids.get(model_name, {})


def tune_model(preprocessor, model_name: str, X_train, y_train, n_splits: int = 5):
    """
    Perform GridSearchCV for the given candidate model name, with the
    preprocessing pipeline included inside the search (fit fresh per
    fold). Returns the fitted GridSearchCV object.
    """
    model = get_candidate_models()[model_name]
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    param_grid = get_param_grid(model_name)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    if not param_grid:
        pipe.fit(X_train, y_train)
        return pipe, {}, None

    search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=skf,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)
