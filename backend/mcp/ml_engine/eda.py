"""`profile_dataset` -- deeper exploratory analysis, on top of `inspection.py`.

Outlier signals (IQR method), pairwise numeric correlations, and --
only if a target column is actually given -- its distribution and, for a
categorical target, class balance. Skips analyses that don't apply
rather than inventing a result: fewer than two numeric columns means no
correlations are returned at all (not an empty-but-present list dressed
up as "no correlation found"); no target column means
`target_distribution`/`class_imbalance` are both `None`, not zeros.
"""

import pandas as pd


def profile_dataset(df: pd.DataFrame, target_column: str | None = None, max_correlations: int = 10) -> dict:
    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    return {
        "outlier_signals": _detect_outliers(df, numeric_columns),
        "top_correlations": _top_correlations(df, numeric_columns, max_correlations),
        "target_distribution": _target_distribution(df, target_column),
        "class_imbalance": _class_imbalance(df, target_column),
    }


def _detect_outliers(df: pd.DataFrame, numeric_columns: list[str]) -> dict:
    """IQR method: flags values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR].

    A simple, standard, deterministic signal -- not a claim that flagged
    points are errors, just that they're numerically unusual relative to
    the rest of that column. Skipped for columns with too few values or
    zero IQR (e.g. a column that's almost all one value).
    """
    outliers = {}
    for col in numeric_columns:
        series = df[col].dropna()
        if len(series) < 4:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        count = int(((series < lower) | (series > upper)).sum())
        if count:
            outliers[col] = {
                "count": count,
                "pct_of_column": round(100 * count / len(series), 2),
                "lower_bound": round(float(lower), 4),
                "upper_bound": round(float(upper), 4),
            }
    return outliers


def _top_correlations(df: pd.DataFrame, numeric_columns: list[str], max_correlations: int) -> list[dict]:
    if len(numeric_columns) < 2:
        return []

    corr_matrix = df[numeric_columns].corr(numeric_only=True)
    pairs = []
    for i, col_a in enumerate(numeric_columns):
        for col_b in numeric_columns[i + 1 :]:
            value = corr_matrix.loc[col_a, col_b]
            if pd.notna(value):
                pairs.append(
                    {"feature_a": col_a, "feature_b": col_b, "correlation": round(float(value), 4)}
                )

    pairs.sort(key=lambda pair: abs(pair["correlation"]), reverse=True)
    return pairs[:max_correlations]


def _is_categorical_like(series: pd.Series, max_unique_for_categorical: int = 20) -> bool:
    non_null = series.dropna()
    if not pd.api.types.is_numeric_dtype(non_null):
        return True
    return non_null.nunique() <= max_unique_for_categorical


def _target_distribution(df: pd.DataFrame, target_column: str | None) -> dict | None:
    if not target_column or target_column not in df.columns:
        return None

    series = df[target_column].dropna()
    if series.empty:
        return None

    if _is_categorical_like(series):
        counts = series.value_counts(dropna=True)
        return {str(k): int(v) for k, v in counts.items()}

    described = series.describe()
    return {stat: round(float(described[stat]), 4) for stat in described.index}


def _class_imbalance(df: pd.DataFrame, target_column: str | None, imbalance_threshold: float = 0.8) -> dict | None:
    """Only meaningful for a categorical-looking target -- `None` for a
    regression-shaped target rather than a nonsensical "imbalance" figure."""
    if not target_column or target_column not in df.columns:
        return None

    series = df[target_column].dropna()
    if series.empty or not _is_categorical_like(series):
        return None

    proportions = series.value_counts(normalize=True, dropna=True)
    if proportions.empty:
        return None

    return {
        "majority_class": str(proportions.idxmax()),
        "majority_proportion": round(float(proportions.max()), 4),
        "minority_class": str(proportions.idxmin()),
        "minority_proportion": round(float(proportions.min()), 4),
        "is_imbalanced": bool(proportions.max() >= imbalance_threshold),
    }
