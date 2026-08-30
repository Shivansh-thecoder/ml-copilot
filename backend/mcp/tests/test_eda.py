import pandas as pd

from ml_engine.eda import profile_dataset


def test_outlier_detection_flags_extreme_values():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 1000]})
    result = profile_dataset(df)
    assert "x" in result["outlier_signals"]
    assert result["outlier_signals"]["x"]["count"] == 1


def test_no_outliers_when_none_present():
    df = pd.DataFrame({"x": [1, 2, 3, 4, 5]})
    result = profile_dataset(df)
    assert result["outlier_signals"] == {}


def test_correlations_skipped_with_fewer_than_two_numeric_columns():
    df = pd.DataFrame({"x": [1, 2, 3], "category": ["a", "b", "c"]})
    result = profile_dataset(df)
    assert result["top_correlations"] == []


def test_correlations_computed_for_two_plus_numeric_columns(regression_df):
    result = profile_dataset(regression_df)
    assert len(result["top_correlations"]) > 0
    pair = result["top_correlations"][0]
    assert "feature_a" in pair and "feature_b" in pair and "correlation" in pair


def test_no_target_means_no_target_distribution(binary_classification_df):
    result = profile_dataset(binary_classification_df, target_column=None)
    assert result["target_distribution"] is None
    assert result["class_imbalance"] is None


def test_class_imbalance_detected(imbalanced_df):
    result = profile_dataset(imbalanced_df, target_column="fraud")
    assert result["class_imbalance"]["is_imbalanced"] is True
    assert result["class_imbalance"]["majority_proportion"] > 0.9


def test_balanced_target_not_flagged_imbalanced(binary_classification_df):
    result = profile_dataset(binary_classification_df, target_column="churned")
    assert result["class_imbalance"] is not None
    # roughly balanced synthetic data -- should not cross the 0.8 threshold
    assert result["class_imbalance"]["is_imbalanced"] is False


def test_regression_target_distribution_is_numeric_summary(regression_df):
    result = profile_dataset(regression_df, target_column="price")
    assert "mean" in result["target_distribution"]
    assert result["class_imbalance"] is None
