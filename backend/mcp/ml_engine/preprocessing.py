"""Building a leakage-safe sklearn preprocessing pipeline.

CRITICAL: this module only ever CONSTRUCTS an unfitted `ColumnTransformer`.
Nothing here calls `.fit()`. Fitting happens exclusively inside
`training.py`/`tuning.py`, as part of a full `sklearn.pipeline.Pipeline`
(`[("preprocessing", ...), ("model", ...)]`), fit only on a training
split or a single cross-validation fold at a time -- never on the whole
dataset up front. That's what prevents leakage: by the time the median
used to impute a numeric column, or the categories an encoder learned,
are computed, the validation/test rows for that split were never looked
at.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str]) -> ColumnTransformer:
    """Return an UNFITTED ColumnTransformer for the given column lists.

    Numeric: median imputation (robust to outliers, unlike mean) then
    standard scaling (needed by LogisticRegression/Ridge; harmless for
    tree-based models, which ignore monotonic rescaling).

    Categorical: most-frequent imputation, then one-hot encoding with
    `handle_unknown="ignore"` -- a category seen only at inference time
    (or only in a held-out fold) becomes an all-zero row instead of
    raising, which matters exactly because this pipeline gets fit
    per-fold/per-split on data that hasn't seen every category yet.
    `sparse_output=False`: keeps output a plain dense array so every
    candidate model (including HistGradientBoosting) can consume it
    identically, without a separate sparse-input code path.
    """
    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if not transformers:
        raise ValueError("No numeric or categorical columns to build a preprocessor for.")

    return ColumnTransformer(transformers)
