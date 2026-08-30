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
