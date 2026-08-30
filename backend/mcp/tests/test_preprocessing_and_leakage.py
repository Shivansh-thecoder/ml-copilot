import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

from ml_engine.preprocessing import build_preprocessor


def test_build_preprocessor_is_unfitted(mixed_types_df):
    preprocessor = build_preprocessor(["numeric_feature"], ["category"])
    assert not hasattr(preprocessor, "transformers_")  # fitted ColumnTransformers gain this attribute


def test_preprocessor_handles_unseen_category_at_transform_time(mixed_types_df):
    """The whole point of handle_unknown='ignore': fit on a subset that
    doesn't contain every category, transform data that does, and confirm
    no exception is raised (leakage-safe CV requires this to work, since
    each fold's training portion may not see every category)."""
    train_subset = mixed_types_df[mixed_types_df["category"] != "C"]
    pipeline = Pipeline(
        [("preprocessing", build_preprocessor(["numeric_feature"], ["category"])), ("model", LogisticRegression())]
    )
    pipeline.fit(train_subset[["numeric_feature", "category"]], train_subset["label"])
    # now predict on data that DOES include the unseen category "C"
    predictions = pipeline.predict(mixed_types_df[["numeric_feature", "category"]])
    assert len(predictions) == len(mixed_types_df)


def test_preprocessing_fit_only_on_training_split_not_full_data():
    """Directly demonstrates the leakage-prevention rule: fit the
    preprocessor on X_train only, and confirm its learned statistic (the
    numeric imputer's median) matches X_train's own median -- and is
    measurably different from the full dataset's median, constructed so
    the two medians CANNOT coincidentally match: X_test is entirely
    large outlier values that would pull the full-data median up if it
    had leaked into fitting.
    """
    import pandas as pd

    X_train = pd.DataFrame({"numeric_feature": [1.0, 2.0, 3.0], "category": ["a", "b", "a"]})
    X_test = pd.DataFrame({"numeric_feature": [1000.0, 1001.0, 1002.0], "category": ["a", "b", "a"]})
    full = pd.concat([X_train, X_test], ignore_index=True)

    preprocessor = build_preprocessor(["numeric_feature"], ["category"])
    preprocessor.fit(X_train)
    median_from_train = preprocessor.named_transformers_["numeric"].named_steps["impute"].statistics_[0]

    assert np.isclose(median_from_train, X_train["numeric_feature"].median())  # == 2.0
    assert not np.isclose(median_from_train, full["numeric_feature"].median())  # full median is ~501.5
