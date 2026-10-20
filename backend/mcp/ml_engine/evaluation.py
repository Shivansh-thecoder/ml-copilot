"""Task-dependent metric computation -- never blindly optimizing accuracy.

Classification and regression have disjoint, non-comparable metric sets
(there's no "accuracy" for a regression target) -- this module keeps them
in separate functions rather than one that branches internally, so
neither task type's metric list can accidentally leak into the other's
output.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    mean_absolute_error,
    make_scorer,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import root_mean_squared_error

# Metrics where a HIGHER value is better -- used by diagnostics.py to know
# which direction "training beats validation" points in, and by
# training.py to rank candidate models correctly.
HIGHER_IS_BETTER = {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "r2"}


def classification_scoring() -> dict:
    """Use the same unweighted macro average in CV, search, and evaluation."""
    return {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, average="macro", pos_label=None, zero_division=0),
        "recall": make_scorer(recall_score, average="macro", pos_label=None, zero_division=0),
        "f1": make_scorer(f1_score, average="macro", pos_label=None, zero_division=0),
    }


def compute_classification_metrics(y_true, y_pred, y_proba=None, *, classes=None) -> dict:
    """Macro metrics give every class equal weight, including string labels.

    Probability columns follow estimator.classes_. For binary AUC metrics,
    classes[1] is the positive class; this ordering is a convention, not an
    inference about which label is semantically positive.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    labels = np.asarray(classes) if classes is not None else np.unique(np.concatenate([y_true, y_pred]))
    if y_proba is not None and len(labels) == 2:
        binary_target = np.asarray(y_true) == labels[1]
        if np.unique(binary_target).size == 2:
            probabilities = np.asarray(y_proba)
            positive_scores = probabilities[:, 1] if probabilities.ndim == 2 else probabilities
            metrics["roc_auc"] = float(roc_auc_score(binary_target, positive_scores))
            metrics["pr_auc"] = float(average_precision_score(binary_target, positive_scores))
    return metrics


def compute_regression_metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(root_mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def choose_primary_metric(task_type: str, class_imbalance: dict | None) -> tuple[str, str]:
    """Pick the ONE metric used to rank/compare models, with a plain-English reason.

    Not "always accuracy": for classification under significant class
    imbalance (see eda.py's `is_imbalanced`), accuracy can look
    deceptively good on a model that just predicts the majority class,
    so macro F1 (which gives each class equal weight) is preferred instead. For regression, RMSE is used --
    penalizes large errors more than MAE, a reasonable general default;
    both are still reported either way (see evaluation output).
    """
    if task_type == "regression":
        return "rmse", "RMSE penalizes large errors more heavily than MAE and is a standard default for regression comparison; MAE and R² are also reported."
    if class_imbalance and class_imbalance.get("is_imbalanced"):
        return (
            "f1",
            "The target classes are imbalanced (majority class "
            f"{class_imbalance.get('majority_proportion', 0):.0%} of samples), so accuracy "
            "would be misleading -- a model predicting only the majority class scores "
            "high on accuracy while being useless. Macro F1 balances precision and recall with equal weight for every class instead.",
        )
    return "f1", "Macro F1 balances precision and recall with equal weight for every class; the same definition is used in CV, tuning, and held-out evaluation. Accuracy is also reported."
