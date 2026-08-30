from ml_engine.inspection import inspect_dataset


def test_shape_and_columns(mixed_types_df):
    result = inspect_dataset(mixed_types_df)
    assert result["n_rows"] == len(mixed_types_df)
    assert result["n_columns"] == 3
    assert set(result["column_names"]) == {"numeric_feature", "category", "label"}


def test_numeric_and_categorical_split(mixed_types_df):
    result = inspect_dataset(mixed_types_df)
    assert "numeric_feature" in result["numeric_columns"]
    assert "category" in result["categorical_columns"]
    assert "category" not in result["numeric_columns"]


def test_missing_value_counts(missing_values_df):
    result = inspect_dataset(missing_values_df)
    assert result["missing_value_counts"]["numeric_feature"] == 10
    assert result["missing_value_counts"]["category"] == 8
    assert "label" not in result["missing_value_counts"]  # no missing values in label


def test_duplicate_row_count():
    import pandas as pd

    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    result = inspect_dataset(df)
    assert result["duplicate_row_count"] == 1


def test_unique_value_counts(mixed_types_df):
    result = inspect_dataset(mixed_types_df)
    assert result["unique_value_counts"]["category"] == 3


def test_numeric_summary_present(mixed_types_df):
    result = inspect_dataset(mixed_types_df)
    assert "numeric_feature" in result["numeric_summary"]
    assert "mean" in result["numeric_summary"]["numeric_feature"]


def test_categorical_top_values_capped(mixed_types_df):
    result = inspect_dataset(mixed_types_df, max_categories_shown=2)
    assert len(result["categorical_top_values"]["category"]) <= 2
