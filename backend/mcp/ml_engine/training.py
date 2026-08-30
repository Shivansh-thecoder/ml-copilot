"""`train_and_compare_models` -- baselines + cross-validation + held-out
evaluation + diagnostics + persistence, as one cohesive step.

One function rather than separate train/cross_validate/evaluate/compare
tools: each MCP tool call starts a fresh server subprocess (see
mcp_client/client.py), so splitting this into multiple calls would mean
multiple subprocess round-trips to do one logical action -- and every
step past the first would need the fitted pipeline from the step before,
which doesn't survive between separate subprocess invocations anyway.

Leakage prevention:
1. `train_test_split` happens FIRST, before anything else touches the data.
2. `build_preprocessor` (preprocessing.py) returns an UNFITTED transformer.
3. Each candidate is wrapped as `Pipeline([("preprocessing", ...), ("model", ...)])`
   and handed whole to `cross_validate` -- sklearn fits a fresh copy of
   the preprocessing step on each fold's OWN training portion; the
   held-out fold is only ever transformed with statistics learned from
   the other folds.
4. The held-out TEST split (from step 1) stays untouched until each
   pipeline's final `.fit(X_train, y_train)` + `.predict(X_test)` --
   preprocessing there is refit on `X_train` only, same rule.
"""

import uuid

from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from .diagnostics import diagnose_fit
from .evaluation import (
    HIGHER_IS_BETTER,
    choose_primary_metric,
    compute_classification_metrics,
    compute_regression_metrics,
)
from .models import get_candidate_models
from .persistence import record_experiment, save_model_artifact
from .preprocessing import build_preprocessor

MIN_SAMPLES_TO_TRAIN = 20


def train_and_compare_models(
    df,
    dataset_id: str,
    dataset_name: str,
    target_column: str,
    task_type: str,
    test_size: float = 0.2,
    cv_folds: int = 5,
    random_state: int = 42,
) -> dict:
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} is not a column in this dataset.")
    if task_type not in ("classification", "regression"):
        raise ValueError(f"task_type must be 'classification' or 'regression', got {task_type!r}.")

    working = df.dropna(subset=[target_column])
    rows_dropped = len(df) - len(working)
    if len(working) < MIN_SAMPLES_TO_TRAIN:
        raise ValueError(
            f"Only {len(working)} rows have a non-missing target value; need at least "
            f"{MIN_SAMPLES_TO_TRAIN} to train/evaluate reliably."
        )

    X = working.drop(columns=[target_column])
    y = working[target_column]

    if task_type == "classification":
        smallest_class_count = int(y.value_counts().min())
        if smallest_class_count < 2:
            raise ValueError(
                "At least one class in the target has fewer than 2 samples; cannot "
                "stratify a train/test split or cross-validation folds."
            )
    effective_cv_folds = min(cv_folds, smallest_class_count) if task_type == "classification" else cv_folds
    if effective_cv_folds < 2:
        raise ValueError("Not enough samples per class to run cross-validation (need at least 2 folds).")

    stratify = y if task_type == "classification" else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    numeric_columns = X.select_dtypes(include="number").columns.tolist()
    categorical_columns = [c for c in X.columns if c not in numeric_columns]

    class_imbalance = None
    if task_type == "classification":
        proportions = y_train.value_counts(normalize=True)
        class_imbalance = {
            "is_imbalanced": bool(proportions.max() >= 0.8),
            "majority_proportion": round(float(proportions.max()), 4),
        }

    primary_metric, metric_reason = choose_primary_metric(task_type, class_imbalance)

    n_classes = int(y_train.nunique()) if task_type == "classification" else None
    scoring = _sklearn_scoring_for(task_type, n_classes)
    cv = (
        StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=random_state)
        if task_type == "classification"
        else KFold(n_splits=effective_cv_folds, shuffle=True, random_state=random_state)
    )

    candidates = get_candidate_models(task_type, random_state=random_state)
    run_id = str(uuid.uuid4())
    results = []

    for model_name, estimator in candidates.items():
        pipeline = Pipeline(
            [
                ("preprocessing", build_preprocessor(numeric_columns, categorical_columns)),
                ("model", estimator),
            ]
        )

        cv_raw = cross_validate(
            pipeline, X_train, y_train, cv=cv, scoring=scoring, return_train_score=True
        )
        cv_mean = {m: _destandardize(m, float(cv_raw[f"test_{m}"].mean())) for m in scoring}
        cv_std = {m: float(cv_raw[f"test_{m}"].std()) for m in scoring}
        train_mean = {m: _destandardize(m, float(cv_raw[f"train_{m}"].mean())) for m in scoring}

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        if task_type == "classification":
            y_proba = pipeline.predict_proba(X_test) if hasattr(pipeline, "predict_proba") else None
            test_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        else:
            test_metrics = compute_regression_metrics(y_test, y_pred)

        diagnostic = diagnose_fit(train_mean[primary_metric], cv_mean[primary_metric], primary_metric)

        artifact_path = save_model_artifact(pipeline, run_id, model_name)

        params = {k: v for k, v in estimator.get_params().items() if not callable(v)}
        record_experiment(
            run_id=run_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            model_name=model_name,
            task_type=task_type,
            target_column=target_column,
            primary_metric=primary_metric,
            cv_mean=cv_mean,
            cv_std=cv_std,
            test_metrics=test_metrics,
            diagnostic=diagnostic["diagnostic"],
            params=params,
            artifact_path=artifact_path,
        )

        results.append(
            {
                "model": model_name,
                "cv_mean": {k: round(v, 4) for k, v in cv_mean.items()},
                "cv_std": {k: round(v, 4) for k, v in cv_std.items()},
                "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
                "diagnostic": diagnostic,
                "artifact_path": str(artifact_path),
            }
        )

    higher_is_better = primary_metric in HIGHER_IS_BETTER
    results.sort(key=lambda r: r["cv_mean"][primary_metric], reverse=higher_is_better)
    best_model = results[0]["model"] if results else None

    return {
        "run_id": run_id,
        "dataset_id": dataset_id,
        "target_column": target_column,
        "task_type": task_type,
        "primary_metric": primary_metric,
        "primary_metric_reason": metric_reason,
        "rows_used": int(len(working)),
        "rows_dropped_missing_target": int(rows_dropped),
        "cv_folds": effective_cv_folds,
        "class_imbalance": class_imbalance,
        "results": results,
        "best_model": best_model,
        "best_model_reasoning": (
            f"'{best_model}' ranked {'highest' if higher_is_better else 'lowest'} on the "
            f"primary metric ({primary_metric}) under {effective_cv_folds}-fold cross-validation "
            f"on the training split -- see 'results' for every candidate's full scores."
            if best_model
            else "No candidate model completed successfully."
        ),
    }


_NEGATED_SCORERS = {"mae", "rmse"}


def _sklearn_scoring_for(task_type: str, n_classes: int | None) -> dict:
    if task_type == "classification":
        scoring = {
            "accuracy": "accuracy",
            "precision": "precision_weighted",
            "recall": "recall_weighted",
            "f1": "f1_weighted",
        }
        if n_classes == 2:
            scoring["roc_auc"] = "roc_auc"
        return scoring
    return {
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
        "r2": "r2",
    }


def _destandardize(metric_name: str, value: float) -> float:
    """sklearn's `neg_*` scorers report negative MAE/RMSE (its convention:
    higher score = better, so error metrics get negated). Flip back to
    the true positive value everywhere this engine reports MAE/RMSE."""
    return -value if metric_name in _NEGATED_SCORERS else value
