import pytest

from ml_engine.training import train_and_compare_models


def test_classification_training_returns_full_comparison(binary_classification_df):
    result = train_and_compare_models(
        binary_classification_df,
        dataset_id="ds-1",
        dataset_name="test",
        target_column="churned",
        task_type="classification",
    )
    assert result["task_type"] == "classification"
    assert result["best_model"] in {"logistic_regression", "random_forest", "hist_gradient_boosting"}
    assert len(result["results"]) == 3
    for candidate in result["results"]:
        assert "cv_mean" in candidate and "cv_std" in candidate
        assert "test_metrics" in candidate
        assert "diagnostic" in candidate
        assert "f1" in candidate["cv_mean"]
        assert "accuracy" in candidate["test_metrics"]


def test_regression_training_returns_full_comparison(regression_df):
    result = train_and_compare_models(
        regression_df,
        dataset_id="ds-2",
        dataset_name="test",
        target_column="price",
        task_type="regression",
    )
    assert result["task_type"] == "regression"
    assert result["primary_metric"] == "rmse"
    for candidate in result["results"]:
        assert "rmse" in candidate["cv_mean"]
        assert "r2" in candidate["test_metrics"]
    # rmse values should be positive (destandardized from sklearn's neg_* convention)
    assert all(c["cv_mean"]["rmse"] > 0 for c in result["results"])


def test_mixed_types_training_handles_categorical(mixed_types_df):
    result = train_and_compare_models(
        mixed_types_df,
        dataset_id="ds-3",
        dataset_name="test",
        target_column="label",
        task_type="classification",
    )
    assert result["best_model"] is not None


def test_missing_values_training_drops_missing_target_rows(missing_values_df):
    # label column itself has no missing values in this fixture, so rows_used == full length
    result = train_and_compare_models(
        missing_values_df,
        dataset_id="ds-4",
        dataset_name="test",
        target_column="label",
        task_type="classification",
    )
    assert result["rows_dropped_missing_target"] == 0
    assert result["rows_used"] == len(missing_values_df)
    # feature-level missing values must not have crashed training (imputed internally)
    assert result["best_model"] is not None


def test_imbalanced_classification_uses_f1_not_accuracy(imbalanced_df):
    result = train_and_compare_models(
        imbalanced_df,
        dataset_id="ds-5",
        dataset_name="test",
        target_column="fraud",
        task_type="classification",
    )
    assert result["class_imbalance"]["is_imbalanced"] is True
    assert result["primary_metric"] == "f1"
    assert "imbalanced" in result["primary_metric_reason"].lower()


def test_missing_target_column_raises(binary_classification_df):
    with pytest.raises(ValueError):
        train_and_compare_models(
            binary_classification_df,
            dataset_id="ds-6",
            dataset_name="test",
            target_column="does_not_exist",
            task_type="classification",
        )


def test_too_few_samples_raises():
    import pandas as pd

    tiny_df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [0, 1, 0, 1, 0]})
    with pytest.raises(ValueError):
        train_and_compare_models(
            tiny_df, dataset_id="ds-7", dataset_name="test", target_column="y", task_type="classification"
        )


def test_invalid_task_type_raises(binary_classification_df):
    with pytest.raises(ValueError):
        train_and_compare_models(
            binary_classification_df,
            dataset_id="ds-8",
            dataset_name="test",
            target_column="churned",
            task_type="not_a_real_task",
        )


def test_best_model_ranked_by_cv_not_train_score(regression_df):
    result = train_and_compare_models(
        regression_df,
        dataset_id="ds-9",
        dataset_name="test",
        target_column="price",
        task_type="regression",
    )
    # lower RMSE is better -- confirm results are sorted ascending by cv_mean rmse
    rmse_values = [c["cv_mean"]["rmse"] for c in result["results"]]
    assert rmse_values == sorted(rmse_values)


def test_string_labels_train_tune_and_predict(binary_classification_df):
    import joblib
    from ml_engine.tuning import tune_model

    df = binary_classification_df.copy()
    df['churned'] = df['churned'].map({0: 'no', 1: 'yes'})
    args = dict(dataset_id='strings', dataset_name='strings', target_column='churned', task_type='classification', cv_folds=3)
    baseline = train_and_compare_models(df, **args)
    tuned = tune_model(df, **args, model_name='logistic_regression', n_iter=2)
    for result in [*baseline['results'], tuned]:
        assert {'f1', 'roc_auc', 'pr_auc'} <= result['test_metrics'].keys()
        model = joblib.load(result['artifact_path'])
        assert set(model.predict(df.drop(columns='churned'))) <= {'no', 'yes'}
