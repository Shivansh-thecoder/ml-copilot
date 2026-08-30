"""MCP tools: `train_models`, `tune_model`.

Both tools resolve dataset_id -> DataFrame the same way as
`dataset_tools.py`/`target_tools.py`, then delegate to `ml_engine.training`
/ `ml_engine.tuning`. Neither tool trains anything if the target is
ambiguous -- both reuse `detect_target` internally and refuse to guess.
"""

from ml_engine.dataset_io import (
    DatasetFileMissingError,
    DatasetNotFoundError,
    EmptyDatasetError,
    UnsupportedDatasetFileError,
    load_dataframe,
)
from ml_engine.models import SUPPORTED_TASK_TYPES, get_candidate_models
from ml_engine.target_detection import detect_target as _detect_target
from ml_engine.training import train_and_compare_models
from ml_engine.tuning import tune_model as _tune_model


def _load_dataset_or_error(dataset_id: str):
    try:
        return load_dataframe(dataset_id), None
    except DatasetNotFoundError as exc:
        return None, {"error": "dataset_not_found", "message": str(exc)}
    except DatasetFileMissingError as exc:
        return None, {"error": "dataset_file_missing", "message": str(exc)}
    except UnsupportedDatasetFileError as exc:
        return None, {"error": "unsupported_file_type", "message": str(exc)}
    except EmptyDatasetError as exc:
        return None, {"error": "empty_dataset", "message": str(exc)}


def _resolve_target_and_task(df, record, target_column: str, task_type: str):
    """Shared by train_models/tune_model: resolve target/task, or hand back
    a needs_clarification result to return as-is (never guess)."""
    if target_column and task_type:
        if task_type not in SUPPORTED_TASK_TYPES:
            return None, None, {"error": "invalid_task_type", "message": f"task_type must be one of {SUPPORTED_TASK_TYPES}"}
        return target_column, task_type, None

    explicit_target = target_column or record.target_column or None
    detection = _detect_target(df, target_column=explicit_target)
    if detection["status"] != "resolved":
        return None, None, detection  # needs_clarification -- returned to caller as-is

    return detection["target_column"], task_type or detection["task_type"], None


def train_models(
    dataset_id: str,
    target_column: str = "",
    task_type: str = "",
    test_size: float = 0.2,
    cv_folds: int = 5,
) -> dict:
    """Train and compare baseline models for a dataset: builds a leakage-safe
    preprocessing pipeline, fits a small set of baseline models appropriate to
    the task (classification or regression), cross-validates each on the
    training split, evaluates each on a held-out test split, runs overfitting/
    underfitting diagnostics, and ranks them by an appropriate metric.

    If the target column is ambiguous and not explicitly given, this returns
    a "needs_clarification" result with candidate columns instead of training
    anything -- call `detect_target` first, or pass `target_column` explicitly,
    to resolve that before calling this.

    Args:
        dataset_id: The id of an already-uploaded dataset.
        target_column: Target column name. If empty, uses the dataset's stored
            target_column, or attempts confident automatic detection.
        task_type: "classification" or "regression". If empty, inferred from
            the target column.
        test_size: Fraction of data held out for final testing (default 0.2).
        cv_folds: Number of cross-validation folds (default 5).
    """
    loaded, error = _load_dataset_or_error(dataset_id)
    if error:
        return error
    df, record = loaded

    resolved_target, resolved_task, error = _resolve_target_and_task(df, record, target_column, task_type)
    if error:
        return error

    try:
        return train_and_compare_models(
            df,
            dataset_id=dataset_id,
            dataset_name=record.name,
            target_column=resolved_target,
            task_type=resolved_task,
            test_size=test_size,
            cv_folds=cv_folds,
        )
    except ValueError as exc:
        return {"error": "training_failed", "message": str(exc)}


def tune_model(
    dataset_id: str,
    model_name: str,
    target_column: str = "",
    task_type: str = "",
    n_iter: int = 15,
    cv_folds: int = 5,
    test_size: float = 0.2,
) -> dict:
    """Run a bounded RandomizedSearchCV hyperparameter search for one specific
    model on a dataset (call `train_models` first to see which model performed
    best, then tune that one).

    Args:
        dataset_id: The id of an already-uploaded dataset.
        model_name: Which model to tune, e.g. "random_forest",
            "hist_gradient_boosting", "logistic_regression"/"ridge".
        target_column: Target column name. If empty, uses the dataset's stored
            target_column, or attempts confident automatic detection.
        task_type: "classification" or "regression". If empty, inferred from
            the target column.
        n_iter: Number of randomly sampled hyperparameter combinations to try
            (bounded search, not exhaustive; default 15).
        cv_folds: Number of cross-validation folds (default 5).
        test_size: Fraction of data held out for final testing (default 0.2).
    """
    loaded, error = _load_dataset_or_error(dataset_id)
    if error:
        return error
    df, record = loaded

    resolved_target, resolved_task, error = _resolve_target_and_task(df, record, target_column, task_type)
    if error:
        return error

    try:
        candidates = get_candidate_models(resolved_task)
    except ValueError as exc:
        return {"error": "invalid_task_type", "message": str(exc)}
    if model_name not in candidates:
        return {
            "error": "unknown_model",
            "message": f"'{model_name}' is not a known model for task_type '{resolved_task}'. Options: {list(candidates)}",
        }

    try:
        return _tune_model(
            df,
            dataset_id=dataset_id,
            dataset_name=record.name,
            target_column=resolved_target,
            task_type=resolved_task,
            model_name=model_name,
            n_iter=n_iter,
            cv_folds=cv_folds,
            test_size=test_size,
        )
    except ValueError as exc:
        return {"error": "tuning_failed", "message": str(exc)}
