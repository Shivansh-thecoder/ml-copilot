"""The candidate model registry -- a small, fixed set per task type.

Deliberately small (three per task): a robust baseline the ML engine
compares before anything fancier. Not an AutoML framework -- adding a
fourth model here is one dict entry, but the scope is a robust baseline,
not dozens of algorithms.
"""

from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge

SUPPORTED_TASK_TYPES = ("classification", "regression")


def get_candidate_models(task_type: str, random_state: int = 42) -> dict:
    """Return {model_name: unfitted estimator} for the given task type.

    Every estimator here is deterministic given `random_state` -- same
    data in, same model out, run to run. `LogisticRegression`/`Ridge` are
    the interpretable linear baseline; `RandomForest*` and
    `HistGradientBoosting*` are two different non-linear baselines
    (bagged trees vs. boosted trees) so the comparison isn't just
    "linear vs. one flavor of tree ensemble."
    """
    if task_type == "classification":
        return {
            "logistic_regression": LogisticRegression(max_iter=1000, random_state=random_state),
            "random_forest": RandomForestClassifier(n_estimators=200, random_state=random_state),
            "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=random_state),
        }
    if task_type == "regression":
        return {
            "ridge": Ridge(random_state=random_state),
            "random_forest": RandomForestRegressor(n_estimators=200, random_state=random_state),
            "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=random_state),
        }
    raise ValueError(f"Unknown task_type {task_type!r}; expected one of {SUPPORTED_TASK_TYPES}")


def get_model_search_space(model_name: str, task_type: str) -> dict:
    """Bounded hyperparameter search space for `tuning.py`'s RandomizedSearchCV.

    Every range here is intentionally small -- this is a controlled,
    bounded search, not an exhaustive one. Prefixed `model__` because
    `tuning.py` searches over the full `Pipeline` (preprocessing + model),
    and sklearn's parameter naming for a `Pipeline` step is
    `<step_name>__<param_name>`.
    """
    spaces: dict[str, dict] = {
        "logistic_regression": {
            "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "ridge": {
            "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
        },
        "random_forest": {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4],
        },
        "hist_gradient_boosting": {
            "model__max_iter": [100, 200, 400],
            "model__max_depth": [None, 5, 10],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
        },
    }
    if model_name not in spaces:
        raise ValueError(f"No search space defined for model {model_name!r}")
    return spaces[model_name]
