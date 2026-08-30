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


def compute_classification_metrics(y_true, y_pred, y_proba=None) -> dict:
    """Accuracy/precision/recall/F1 always; ROC-AUC/PR-AUC only for binary
    targets with probability estimates available -- both require a
    single "positive class" score, which isn't well-defined the same way
    for multiclass without further choices this engine doesn't make
    silently. `average="binary"` for 2 classes (the standard,
    unambiguous choice), `"macro"` for more (unweighted mean across
    classes, so a rare class isn't drowned out by a common one)."""
    classes_seen = set(y_true) | set(y_pred)
    average = "binary" if len(classes_seen) <= 2 else "macro"

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
    }

    if y_proba is not None and len(classes_seen) == 2:
        positive_scores = y_proba[:, 1] if getattr(y_proba, "ndim", 1) == 2 else y_proba
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, positive_scores))
            metrics["pr_auc"] = float(average_precision_score(y_true, positive_scores))
        except ValueError:
            # e.g. y_true has only one class present in this particular
            # split -- ROC-AUC is undefined; omit rather than fabricate.
            pass

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
    so F1 (which accounts for both precision and recall on the minority
    class) is preferred instead. For regression, RMSE is used --
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
            "high on accuracy while being useless. F1 balances precision and recall instead.",
        )
    return "f1", "F1 balances precision and recall and is a reasonable default for classification comparison; accuracy is also reported."
