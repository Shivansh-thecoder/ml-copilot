import pytest

from ml_engine.tuning import tune_model


def test_tune_classification_model_returns_best_params(binary_classification_df):
    result = tune_model(
        binary_classification_df,
        dataset_id="ds-1",
        dataset_name="test",
        target_column="churned",
        task_type="classification",
        model_name="logistic_regression",
        n_iter=5,
        cv_folds=3,
    )
    assert "best_params" in result
    assert "model__C" in result["best_params"]
    assert result["best_cv_score"] is not None
    assert result["search_configuration"]["n_iter"] <= 5


def test_tune_regression_model_returns_positive_rmse(regression_df):
    result = tune_model(
        regression_df,
        dataset_id="ds-2",
        dataset_name="test",
        target_column="price",
        task_type="regression",
        model_name="ridge",
        n_iter=5,
        cv_folds=3,
    )
    assert result["test_metrics"]["rmse"] > 0
    # best_cv_score should also be positive after destandardization
    assert result["best_cv_score"] > 0


def test_tune_bounds_n_iter_to_search_space_size(binary_classification_df):
    # logistic_regression's search space has exactly 5 values for model__C
    result = tune_model(
        binary_classification_df,
        dataset_id="ds-3",
        dataset_name="test",
        target_column="churned",
        task_type="classification",
        model_name="logistic_regression",
        n_iter=1000,  # deliberately huge -- must not exceed available combos
        cv_folds=3,
    )
    assert result["search_configuration"]["n_iter"] == 5


@pytest.mark.parametrize("task_type", ["classification", "regression"])
def test_diagnostic_uses_selected_candidate_fold_scores(
    task_type, binary_classification_df, regression_df, monkeypatch
):
    import ml_engine.tuning as tuning

    classification = task_type == "classification"
    train_score, cv_score = (0.95, 0.7) if classification else (-1.0, -2.0)

    class ControlledSearch:
        def __init__(self, estimator, **kwargs):
            assert kwargs["return_train_score"] is True
            self.best_estimator_ = estimator
            self.best_index_ = 1
            self.best_score_ = cv_score
            self.best_params_ = {}
            self.cv_results_ = {
                "mean_train_score": [0.0, train_score],
                "mean_test_score": [0.0, cv_score],
            }

        def fit(self, X, y):
            self.best_estimator_.fit(X, y)

    monkeypatch.setattr(tuning, "RandomizedSearchCV", ControlledSearch)
    # Holdout performance must not become either diagnostic input.
    metric = "f1" if classification else "rmse"
    evaluator = "compute_classification_metrics" if classification else "compute_regression_metrics"
    monkeypatch.setattr(tuning, evaluator, lambda *args, **kwargs: {metric: 0.1234})
    result = tune_model(
        binary_classification_df if classification else regression_df,
        dataset_id="diagnostic-test",
        dataset_name="test",
        target_column="churned" if classification else "price",
        task_type=task_type,
        model_name="logistic_regression" if classification else "ridge",
        n_iter=2,
        cv_folds=3,
    )
    assert result["diagnostic"]["train_score"] == abs(train_score)
    assert result["diagnostic"]["validation_score"] == abs(cv_score)
    assert result["diagnostic"]["diagnostic"] == "possible_overfitting"
    assert result["best_cv_score"] == abs(cv_score)
    assert result["test_metrics"][metric] == 0.1234
