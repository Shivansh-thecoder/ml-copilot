"""MCP tool: `detect_target`.

See `ml_engine/target_detection.py` for the actual logic -- this file
only adds dataset-id resolution and the "use the Dataset row's own
target_column if the caller didn't override it" rule.
"""

from ml_engine.dataset_io import (
    DatasetFileMissingError,
    DatasetNotFoundError,
    EmptyDatasetError,
    UnsupportedDatasetFileError,
    load_dataframe,
)
from ml_engine.target_detection import detect_target as _detect_target


def detect_target(dataset_id: str, target_column: str = "") -> dict:
    """Determine the target column and ML task type (classification or regression)
    for a dataset.

    If `target_column` is given, or the dataset already has one set, it is
    used directly with no guessing. Otherwise this analyzes column names and
    statistics to find a confident candidate. If no column is a clear, confident
    match, the result's `status` is "needs_clarification" with candidate columns
    and reasoning -- ask the user which column to use rather than assuming.

    Args:
        dataset_id: The id of an already-uploaded dataset.
        target_column: Optional explicit target column name. Overrides the
            dataset's stored target_column if both are present.
    """
    try:
        df, record = load_dataframe(dataset_id)
    except DatasetNotFoundError as exc:
        return {"error": "dataset_not_found", "message": str(exc)}
    except DatasetFileMissingError as exc:
        return {"error": "dataset_file_missing", "message": str(exc)}
    except UnsupportedDatasetFileError as exc:
        return {"error": "unsupported_file_type", "message": str(exc)}
    except EmptyDatasetError as exc:
        return {"error": "empty_dataset", "message": str(exc)}

    resolved_target = target_column or record.target_column or None
    return _detect_target(df, target_column=resolved_target)
