"""MCP tools: `inspect_dataset`, `profile_dataset`.

Plain, undecorated functions that `server.py` registers with
`mcp.tool()`. Each function here does exactly three things -- load the
dataset by id (`dataset_io`),
call the corresponding `ml_engine` function, and turn any known failure
into a clear, structured `{"error": ...}` result rather than letting an
exception cross the MCP boundary as an opaque protocol error.
"""

from ml_engine.dataset_io import (
    DatasetFileMissingError,
    DatasetNotFoundError,
    EmptyDatasetError,
    UnsupportedDatasetFileError,
    load_dataframe,
)
from ml_engine.eda import profile_dataset as _profile_dataset
from ml_engine.inspection import inspect_dataset as _inspect_dataset


def _load_or_error(dataset_id: str):
    """Shared error handling for every tool that starts from a dataset_id.

    Returns `(df, record, None)` on success or `(None, None, error_dict)`
    on a known, expected failure -- callers check which they got rather
    than catching exceptions themselves, keeping each tool function's
    body focused on its own logic.
    """
    try:
        df, record = load_dataframe(dataset_id)
        return df, record, None
    except DatasetNotFoundError as exc:
        return None, None, {"error": "dataset_not_found", "message": str(exc)}
    except DatasetFileMissingError as exc:
        return None, None, {"error": "dataset_file_missing", "message": str(exc)}
    except UnsupportedDatasetFileError as exc:
        return None, None, {"error": "unsupported_file_type", "message": str(exc)}
    except EmptyDatasetError as exc:
        return None, None, {"error": "empty_dataset", "message": str(exc)}


def inspect_dataset(dataset_id: str) -> dict:
    """Inspect a dataset's structure: shape, dtypes, missing values, duplicates,
    numeric/categorical columns, unique-value counts, and basic numeric statistics.

    Use this first, before any other analysis, to understand what a
    dataset actually contains.

    Args:
        dataset_id: The id of an already-uploaded dataset.
    """
    df, _record, error = _load_or_error(dataset_id)
    if error:
        return error
    return _inspect_dataset(df)


def profile_dataset(dataset_id: str, target_column: str = "") -> dict:
    """Run exploratory data analysis on a dataset: outlier signals (IQR method),
    the strongest pairwise correlations between numeric columns, and -- if a
    target column is given -- its distribution and class balance.

    Use this after `inspect_dataset` for a deeper look, especially before
    training models, to understand feature relationships and target imbalance.

    Args:
        dataset_id: The id of an already-uploaded dataset.
        target_column: Optional target column name to profile the distribution
            of. Leave empty to skip target-specific analysis.
    """
    df, record, error = _load_or_error(dataset_id)
    if error:
        return error
    resolved_target = target_column or record.target_column or None
    return _profile_dataset(df, target_column=resolved_target)
