from ml_engine.target_detection import detect_target


def test_explicit_target_used_directly(binary_classification_df):
    result = detect_target(binary_classification_df, target_column="churned")
    assert result["status"] == "resolved"
    assert result["target_column"] == "churned"
    assert result["task_type"] == "classification"


def test_explicit_invalid_target_needs_clarification(binary_classification_df):
    result = detect_target(binary_classification_df, target_column="does_not_exist")
    assert result["status"] == "needs_clarification"
    assert "does_not_exist" in result["reasoning"]


def test_explicit_target_infers_regression(regression_df):
    result = detect_target(regression_df, target_column="price")
    assert result["task_type"] == "regression"


def test_confident_auto_detection_by_name(binary_classification_df):
    result = detect_target(binary_classification_df, target_column=None)
    assert result["status"] == "resolved"
    assert result["target_column"] == "churned"


def test_ambiguous_dataset_needs_clarification(ambiguous_target_df):
    result = detect_target(ambiguous_target_df, target_column=None)
    assert result["status"] == "needs_clarification"
    assert isinstance(result["candidates"], list)


def test_never_returns_target_without_reasoning(ambiguous_target_df):
    result = detect_target(ambiguous_target_df)
    assert result["reasoning"]
    assert result["target_column"] is None


def test_all_constant_columns_needs_clarification():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 1], "b": [2, 2, 2]})
    result = detect_target(df)
    assert result["status"] == "needs_clarification"
