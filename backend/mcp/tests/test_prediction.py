import pytest
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline

from ml_engine.persistence import record_experiment, save_best_model, save_model_artifact
from ml_engine.prediction import (
    InvalidFeatureValueError,
    MissingFeaturesError,
    NoSavedModelError,
    describe_saved_model,
    predict_with_saved_model,
)
from ml_engine.preprocessing import build_preprocessor


def _save_pipeline(df, feature_columns, target, estimator, task_type, dataset_id):
    """Fit a pipeline the same way training.py does, then designate it as
    this dataset's saved model -- the exact artifact prediction reloads."""
    numeric = df[feature_columns].select_dtypes(include="number").columns.tolist()
    categorical = [c for c in feature_columns if c not in numeric]
    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessor(numeric, categorical)),
            ("model", estimator),
        ]
    )
    pipeline.fit(df[feature_columns], df[target])

    artifact_path = save_model_artifact(pipeline, run_id="run-p", model_name="m")
    record_experiment(
        run_id="run-p",
        dataset_id=dataset_id,
        dataset_name="test",
        model_name="m",
        task_type=task_type,
        target_column=target,
        primary_metric="f1" if task_type == "classification" else "rmse",
        cv_mean={},
        cv_std={},
        test_metrics={},
        diagnostic="no_strong_signal",
        params={},
        artifact_path=artifact_path,
    )
    save_best_model(run_id="run-p", model_name="m", dataset_id=dataset_id)
    return pipeline


@pytest.fixture
def saved_classifier(mixed_types_df):
    _save_pipeline(
        mixed_types_df,
        ["numeric_feature", "category"],
        "label",
        LogisticRegression(max_iter=1000),
        "classification",
        "ds-clf",
    )
    return "ds-clf"


@pytest.fixture
def saved_regressor(regression_df):
    _save_pipeline(
        regression_df, ["feature_1", "feature_2"], "price", Ridge(), "regression", "ds-reg"
    )
    return "ds-reg"


def test_schema_comes_from_the_saved_pipeline(saved_classifier):
    schema = describe_saved_model(saved_classifier)

    assert [f["name"] for f in schema["features"]] == ["numeric_feature", "category"]
    kinds = {f["name"]: f["kind"] for f in schema["features"]}
    assert kinds == {"numeric_feature": "numeric", "category": "categorical"}

    categorical = next(f for f in schema["features"] if f["name"] == "category")
    assert sorted(categorical["categories"]) == ["A", "B", "C"]
    assert schema["classes"] == [0, 1]
    assert schema["supports_probabilities"] is True
    assert schema["target_column"] == "label"


def test_schema_for_regression_has_no_classes(saved_regressor):
    schema = describe_saved_model(saved_regressor)
    assert schema["classes"] is None
    assert schema["supports_probabilities"] is False


def test_schema_without_a_saved_model_raises():
    with pytest.raises(NoSavedModelError):
        describe_saved_model("never-saved")


def test_classification_prediction_includes_probabilities(saved_classifier):
    result = predict_with_saved_model(
        saved_classifier, [{"numeric_feature": 1.5, "category": "A"}]
    )

    prediction = result["predictions"][0]
    assert prediction["prediction"] in (0, 1)
    assert set(prediction["probabilities"]) == {"0", "1"}
    assert prediction["probabilities"]["0"] + prediction["probabilities"]["1"] == pytest.approx(1.0)
    assert prediction["confidence"] == max(prediction["probabilities"].values())


def test_regression_prediction_returns_a_number(saved_regressor):
    result = predict_with_saved_model(saved_regressor, [{"feature_1": 1.0, "feature_2": -1.0}])

    prediction = result["predictions"][0]
    assert isinstance(prediction["prediction"], float)
    assert "probabilities" not in prediction
    assert result["classes"] is None


def test_predicts_multiple_rows_at_once(saved_regressor):
    result = predict_with_saved_model(
        saved_regressor,
        [{"feature_1": 1.0, "feature_2": 0.0}, {"feature_1": -2.0, "feature_2": 3.0}],
    )
    assert [p["row"] for p in result["predictions"]] == [0, 1]
    assert result["predictions"][0]["prediction"] != result["predictions"][1]["prediction"]


def test_string_values_are_coerced_for_numeric_features(saved_regressor):
    """The frontend sends form input as strings; numeric columns coerce."""
    from_strings = predict_with_saved_model(
        saved_regressor, [{"feature_1": "1.0", "feature_2": "-1.0"}]
    )
    from_numbers = predict_with_saved_model(saved_regressor, [{"feature_1": 1.0, "feature_2": -1.0}])
    assert from_strings["predictions"][0]["prediction"] == from_numbers["predictions"][0]["prediction"]


def test_missing_feature_raises_with_the_column_named(saved_classifier):
    with pytest.raises(MissingFeaturesError) as exc_info:
        predict_with_saved_model(saved_classifier, [{"numeric_feature": 1.0}])
    assert exc_info.value.missing == ["category"]
    assert exc_info.value.row_index == 0


def test_blank_value_counts_as_missing(saved_classifier):
    with pytest.raises(MissingFeaturesError) as exc_info:
        predict_with_saved_model(saved_classifier, [{"numeric_feature": "  ", "category": "A"}])
    assert exc_info.value.missing == ["numeric_feature"]


def test_non_numeric_value_for_a_numeric_feature_raises(saved_classifier):
    with pytest.raises(InvalidFeatureValueError) as exc_info:
        predict_with_saved_model(
            saved_classifier, [{"numeric_feature": "warm", "category": "A"}]
        )
    assert exc_info.value.column == "numeric_feature"


def test_extra_columns_are_ignored_not_rejected(saved_classifier):
    """The target column is the common case -- a user pasting a full row
    shouldn't be an error."""
    result = predict_with_saved_model(
        saved_classifier, [{"numeric_feature": 1.0, "category": "A", "label": 1}]
    )
    assert result["ignored_columns"] == ["label"]
    assert result["predictions"][0]["prediction"] in (0, 1)


def test_unseen_category_warns_but_still_predicts(saved_classifier):
    result = predict_with_saved_model(
        saved_classifier, [{"numeric_feature": 1.0, "category": "Z"}]
    )
    assert result["predictions"][0]["prediction"] in (0, 1)
    assert any("category" in warning for warning in result["warnings"])


def test_prediction_without_a_saved_model_raises():
    with pytest.raises(NoSavedModelError):
        predict_with_saved_model("never-saved", [{"anything": 1}])


def test_empty_rows_rejected(saved_classifier):
    with pytest.raises(ValueError):
        predict_with_saved_model(saved_classifier, [])


def test_prediction_reuses_the_training_preprocessing(saved_regressor, regression_df):
    """The saved pipeline's own scaler/imputer must be what transforms new
    rows -- so a single row predicts identically whether it arrives alone
    or inside a batch with wildly different values. Refitting preprocessing
    on the incoming rows would make the batch shift this row's result."""
    alone = predict_with_saved_model(saved_regressor, [{"feature_1": 0.5, "feature_2": 0.5}])
    in_batch = predict_with_saved_model(
        saved_regressor,
        [{"feature_1": 0.5, "feature_2": 0.5}, {"feature_1": 900.0, "feature_2": -900.0}],
    )
    assert alone["predictions"][0]["prediction"] == in_batch["predictions"][0]["prediction"]
