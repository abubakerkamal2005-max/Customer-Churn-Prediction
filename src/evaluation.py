"""
src/evaluation.py
==================
Model evaluation utilities: metric computation, cross-validation scoring,
and human-readable reporting. Accuracy alone is intentionally never used
as the sole selection criterion because the target is imbalanced
(~73% No-Churn / ~27% Churn in the Telco dataset) — a model that always
predicts "No" would score ~73% accuracy while being useless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Why each metric matters for churn prediction:
METRIC_RATIONALE = {
    "accuracy": "Overall correctness, but misleading alone under class imbalance.",
    "precision": "Of customers flagged as churn risks, how many actually churn — "
    "matters because retention offers cost money; low precision wastes budget.",
    "recall": "Of customers who actually churn, how many the model catches — "
    "matters because missed churners are lost revenue with no chance to intervene.",
    "f1": "Harmonic mean of precision and recall; a single balanced summary "
    "when both false positives and false negatives carry real cost.",
    "roc_auc": "Threshold-independent measure of how well the model ranks "
    "churners above non-churners; robust to the exact decision threshold chosen.",
    "pr_auc": "Like ROC-AUC but focused on the positive (churn) class; more "
    "informative than ROC-AUC under class imbalance.",
}


@dataclass
class EvaluationResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    confusion_matrix: list

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "roc_auc": round(self.roc_auc, 4),
            "pr_auc": round(self.pr_auc, 4),
            "confusion_matrix": self.confusion_matrix,
        }


def evaluate_predictions(
    model_name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray | None
) -> EvaluationResult:
    """Compute the full metric suite for a single model's held-out predictions."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_proba is not None:
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
    else:
        roc_auc = float("nan")
        pr_auc = float("nan")

    cm = confusion_matrix(y_true, y_pred).tolist()

    return EvaluationResult(
        model_name=model_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        f1=f1,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        confusion_matrix=cm,
    )


def format_comparison_table(results: list[EvaluationResult]) -> str:
    """Render a plain-text comparison table for the training summary/logs."""
    header = f"{'Model':<28}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}{'ROC-AUC':>9}{'PR-AUC':>8}"
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.model_name:<28}{r.accuracy:>10.4f}{r.precision:>11.4f}"
            f"{r.recall:>9.4f}{r.f1:>8.4f}{r.roc_auc:>9.4f}{r.pr_auc:>8.4f}"
        )
    return "\n".join(lines)
