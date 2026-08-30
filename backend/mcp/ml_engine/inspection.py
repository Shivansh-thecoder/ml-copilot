"""`inspect_dataset` -- structural facts about a DataFrame, compactly.

Deliberately the "what IS this data, mechanically" layer -- shape,
dtypes, missingness, duplicates, numeric/categorical split, unique
counts, basic numeric stats. `eda.py`'s `profile_dataset` builds on top
of this for deeper analysis (correlations, outliers, target/class
balance); this module doesn't need a target column at all.

Every function here returns a plain, bounded dict -- never the raw
DataFrame, and never an unbounded per-row structure. A 500-column
dataset produces a 500-entry dtype dict (still bounded, one line per
column) but `categorical_top_values` is capped at `max_categories_shown`
per column specifically so a high-cardinality text column can't balloon
the result -- this is what "avoid dumping enormous dataframes into the
LLM context" means in practice.
"""

import math

import pandas as pd


def inspect_dataset(df: pd.DataFrame, max_categories_shown: int = 10) -> dict:
    """Compute structural facts about `df`. Read-only; never mutates `df`."""
    numeric_columns = df.select_dtypes(include="number").columns.tolist()
    categorical_columns = [c for c in df.columns if c not in numeric_columns]

    missing = df.isna().sum()
    missing_value_counts = {col: int(missing[col]) for col in df.columns if missing[col] > 0}

    unique_value_counts = {col: int(df[col].nunique(dropna=True)) for col in df.columns}

    numeric_summary = {}
    if numeric_columns:
        described = df[numeric_columns].describe()
        numeric_summary = {
            col: {stat: _safe_float(described.loc[stat, col]) for stat in described.index}
            for col in numeric_columns
        }

    categorical_top_values = {}
    for col in categorical_columns:
        counts = df[col].value_counts(dropna=True).head(max_categories_shown)
        categorical_top_values[col] = {str(k): int(v) for k, v in counts.items()}

    return {
        "n_rows": int(df.shape[0]),
        "n_columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "missing_value_counts": missing_value_counts,
        "duplicate_row_count": int(df.duplicated().sum()),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "unique_value_counts": unique_value_counts,
        "numeric_summary": numeric_summary,
        "categorical_top_values": categorical_top_values,
    }


def _safe_float(value) -> float | None:
    """`describe()` can produce NaN (e.g. std of a single-row column) --
    NaN isn't valid JSON, so it becomes `None` rather than a string that
    downstream code would have to special-case."""
    if value is None:
        return None
    as_float = float(value)
    return None if math.isnan(as_float) else as_float
