"""`tune_model` -- controlled, bounded hyperparameter search for one chosen model.

`RandomizedSearchCV`, not `GridSearchCV`: samples a bounded number of
combinations (`n_iter`) from the search space (`models.py`'s
`get_model_search_space`) instead of trying every combination
exhaustively -- runtime stays predictable regardless of how many
hyperparameters a model has, at the cost of not being guaranteed to find
the single best combination in the space. That tradeoff is the point:
"a controlled tuning strategy... do not create an enormous computational
search."

Stateless like `training.py`: takes a DataFrame and a model name, does
its own train/test split and preprocessing from scratch (same
`random_state` default as `training.py`, so the same call with the same
arguments reproduces the same split), and persists its own result --
no dependency on a previous `train_and_compare_models` call's in-memory
state, which wouldn't survive to a separate MCP call anyway (see
training.py's module docstring).
"""

import uuid

from sklearn.model_selection import KFold, RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline

from .diagnostics import diagnose_fit
from .evaluation import choose_primary_metric, compute_classification_metrics, compute_regression_metrics
from .models import get_candidate_models, get_model_search_space
from .persistence import record_experiment, save_model_artifact
from .preprocessing import build_preprocessor

DEFAULT_N_ITER = 15


def tune_model(
    df,
    dataset_id: str,
    dataset_name: str,
    target_column: str,
    task_type: str,
    model_name: str,
    n_iter: int = DEFAULT_N_ITER,
    cv_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict:
    if target_column not in df.columns:
        raise ValueError(f"target_column {target_column!r} is not a column in this dataset.")

    candidates = get_candidate_models(task_type, random_state=random_state)
    if model_name not in candidates:
        raise ValueError(f"Unknown model {model_name!r} for task_type {task_type!r}. Options: {list(candidates)}")
    search_space = get_model_search_space(model_name, task_type)

    working = df.dropna(subset=[target_column])
    X = working.drop(columns=[target_column])
    y = working[target_column]

    if task_type == "classification":
        smallest_class_count = int(y.value_counts().min())
        if smallest_class_count < 2:
            raise ValueError("At least one class has fewer than 2 samples; cannot stratify a split.")
        effective_cv_folds = min(cv_folds, smallest_class_count)
    else:
        effective_cv_folds = cv_folds
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
        class_imbalance = {"is_imbalanced": bool(proportions.max() >= 0.8)}
    primary_metric, metric_reason = choose_primary_metric(task_type, class_imbalance)
    scoring = _single_scorer_for(primary_metric)

    cv = (
        StratifiedKFold(n_splits=effective_cv_folds, shuffle=True, random_state=random_state)
        if task_type == "classification"
        else KFold(n_splits=effective_cv_folds, shuffle=True, random_state=random_state)
    )

    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessor(numeric_columns, categorical_columns)),
            ("model", candidates[model_name]),
        ]
    )

    bounded_n_iter = min(n_iter, _search_space_size(search_space))
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=search_space,
        n_iter=bounded_n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        refit=True,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    y_pred = best_pipeline.predict(X_test)
    if task_type == "classification":
        y_proba = best_pipeline.predict_proba(X_test) if hasattr(best_pipeline, "predict_proba") else None
        test_metrics = compute_classification_metrics(y_test, y_pred, y_proba)
    else:
        test_metrics = compute_regression_metrics(y_test, y_pred)

    # Diagnostic compares the CV-search's best score (its closest analogue
    # to a "training-side" score, since RandomizedSearchCV doesn't expose
    # a plain train score directly) against the held-out test score.
    # sklearn's neg_* scorers report MAE/RMSE negated (its "higher score
    # = better" convention) -- flip back to the true positive value
    # before comparing against test_metrics, which is already positive.
    best_cv_score = float(search.best_score_)
    if primary_metric in ("mae", "rmse"):
        best_cv_score = -best_cv_score
    diagnostic = diagnose_fit(best_cv_score, test_metrics.get(primary_metric, best_cv_score), primary_metric)

    run_id = str(uuid.uuid4())
    artifact_path = save_model_artifact(best_pipeline, run_id, model_name)

    record_experiment(
        run_id=run_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        model_name=model_name,
        task_type=task_type,
        target_column=target_column,
        primary_metric=primary_metric,
        cv_mean={primary_metric: best_cv_score},
        cv_std={},
        test_metrics=test_metrics,
        diagnostic=diagnostic["diagnostic"],
        params=search.best_params_,
        artifact_path=artifact_path,
        status="tuned",
    )

    return {
        "run_id": run_id,
        "model": model_name,
        "primary_metric": primary_metric,
        "primary_metric_reason": metric_reason,
        "best_params": search.best_params_,
        "best_cv_score": round(best_cv_score, 4),
        "test_metrics": {k: round(v, 4) for k, v in test_metrics.items()},
        "diagnostic": diagnostic,
        "search_configuration": {
            "n_iter": bounded_n_iter,
            "cv_folds": effective_cv_folds,
            "search_space": search_space,
        },
        "artifact_path": str(artifact_path),
    }


def _single_scorer_for(metric_name: str) -> str:
    mapping = {
        "accuracy": "accuracy",
        "f1": "f1_weighted",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "roc_auc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2",
    }
    return mapping.get(metric_name, metric_name)


def _search_space_size(search_space: dict) -> int:
    """Total combinations available -- RandomizedSearchCV can't sample
    more distinct combinations than exist, and raises if asked to."""
    size = 1
    for values in search_space.values():
        size *= len(values)
    return size
