from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml_engine.persistence import (
    get_experiment,
    get_saved_model_metadata,
    list_experiments,
    load_model_artifact,
    record_experiment,
    save_best_model,
    save_model_artifact,
)
from ml_engine.preprocessing import build_preprocessor


def _fitted_pipeline(binary_classification_df):
    pipeline = Pipeline(
        [
            ("preprocessing", build_preprocessor(["feature_1", "feature_2"], [])),
            ("model", LogisticRegression()),
        ]
    )
    pipeline.fit(
        binary_classification_df[["feature_1", "feature_2"]], binary_classification_df["churned"]
    )
    return pipeline


def test_save_and_load_artifact_round_trip(binary_classification_df):
    pipeline = _fitted_pipeline(binary_classification_df)
    path = save_model_artifact(pipeline, run_id="run-1", model_name="logistic_regression")
    assert Path(path).exists()

    loaded = load_model_artifact(path)
    predictions = loaded.predict(binary_classification_df[["feature_1", "feature_2"]])
    assert len(predictions) == len(binary_classification_df)


def test_record_and_get_experiment():
    record_experiment(
        run_id="run-2",
        dataset_id="ds-1",
        dataset_name="test",
        model_name="random_forest",
        task_type="classification",
        target_column="churned",
        primary_metric="f1",
        cv_mean={"f1": 0.85},
        cv_std={"f1": 0.02},
        test_metrics={"f1": 0.83, "accuracy": 0.9},
        diagnostic="no_strong_signal",
        params={"n_estimators": 200},
        artifact_path=Path("some/path.joblib"),
    )
    rows = get_experiment("run-2", "random_forest")
    assert len(rows) == 1
    assert rows[0]["cv_mean"]["f1"] == 0.85
    assert rows[0]["test_metrics"]["accuracy"] == 0.9


def test_list_experiments_for_dataset():
    record_experiment(
        run_id="run-3",
        dataset_id="ds-2",
        dataset_name="test2",
        model_name="ridge",
        task_type="regression",
        target_column="price",
        primary_metric="rmse",
        cv_mean={"rmse": 1.2},
        cv_std={"rmse": 0.1},
        test_metrics={"rmse": 1.3, "r2": 0.8},
        diagnostic="no_strong_signal",
        params={},
        artifact_path=Path("some/path2.joblib"),
    )
    runs = list_experiments("ds-2")
    assert len(runs) == 1
    assert runs[0]["model_name"] == "ridge"


def test_save_best_model_and_retrieve_metadata(binary_classification_df):
    pipeline = _fitted_pipeline(binary_classification_df)
    artifact_path = save_model_artifact(pipeline, run_id="run-4", model_name="logistic_regression")
    record_experiment(
        run_id="run-4",
        dataset_id="ds-3",
        dataset_name="test3",
        model_name="logistic_regression",
        task_type="classification",
        target_column="churned",
        primary_metric="f1",
        cv_mean={"f1": 0.9},
        cv_std={"f1": 0.01},
        test_metrics={"f1": 0.88},
        diagnostic="no_strong_signal",
        params={},
        artifact_path=artifact_path,
    )

    result = save_best_model(run_id="run-4", model_name="logistic_regression", dataset_id="ds-3")
    assert Path(result["saved_model_path"]).exists()

    metadata = get_saved_model_metadata("ds-3")
    assert metadata["model_name"] == "logistic_regression"
    assert metadata["run_id"] == "run-4"


def test_no_saved_model_returns_none():
    assert get_saved_model_metadata("no-such-dataset") is None
