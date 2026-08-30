"""Resolving a dataset_id to an actual file, and loading it into a DataFrame.

This is the ONLY module in `ml_engine/` that touches storage (the SQLite
`datasets` table, the uploaded files under `storage/`). Every other
`ml_engine/` module takes a `pandas.DataFrame` directly and has no idea
datasets have ids, rows in a database, or files on disk at all -- see
`__init__.py`'s module docstring for why this file avoids `app.core.config`
and talks to `app.db` with plain `sqlite3` instead.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

# This file lives at backend/mcp/ml_engine/dataset_io.py, so two parents
# up is backend/ -- where app.db and storage/ both live (matching
# app/core/config.py's defaults: database_url="sqlite:///./app.db",
# storage_dir="storage/datasets", both resolved relative to backend/
# when the FastAPI app runs from there).
_BACKEND_DIR = Path(__file__).resolve().parents[2]
_DB_PATH = _BACKEND_DIR / "app.db"

SUPPORTED_EXTENSIONS = {".csv"}


class DatasetNotFoundError(Exception):
    """No row with this id in the `datasets` table (or no database at all)."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"Dataset '{dataset_id}' not found.")


class DatasetFileMissingError(Exception):
    """The database row exists, but its file isn't on disk."""

    def __init__(self, dataset_id: str, path: Path) -> None:
        self.dataset_id = dataset_id
        self.path = path
        super().__init__(f"Dataset '{dataset_id}' has no file at expected path: {path}")


class UnsupportedDatasetFileError(Exception):
    """The file exists but isn't a type this engine can read yet."""

    def __init__(self, path: Path) -> None:
        self.path = path
        super().__init__(
            f"Unsupported dataset file type '{path.suffix or '(none)'}' for {path}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )


class EmptyDatasetError(Exception):
    """The file loaded, but has no rows (or no columns) to work with."""

    def __init__(self, dataset_id: str, reason: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"Dataset '{dataset_id}' is empty: {reason}")


@dataclass
class DatasetRecord:
    id: str
    name: str
    target_column: str  # "" means not set by the user
    stored_path: Path


def get_dataset_record(dataset_id: str) -> DatasetRecord:
    """Look up one dataset's row by id -- direct, read-only SQL, nothing else.

    Deliberately duplicates none of DatasetService's business logic
    (create/update/delete, file-saving, size formatting) -- this reads
    exactly the four columns this engine needs and nothing more.
    """
    if not _DB_PATH.exists():
        raise DatasetNotFoundError(dataset_id)

    conn = sqlite3.connect(str(_DB_PATH))
    try:
        cursor = conn.execute(
            "SELECT id, name, target_column, stored_path FROM datasets WHERE id = ?",
            (dataset_id,),
        )
        row = cursor.fetchone()
    finally:
        conn.close()

    if row is None:
        raise DatasetNotFoundError(dataset_id)

    row_id, name, target_column, stored_path = row
    return DatasetRecord(
        id=row_id,
        name=name,
        target_column=target_column or "",
        stored_path=Path(stored_path),
    )


def resolve_file_path(record: DatasetRecord) -> Path:
    """Turn a (possibly relative) stored_path into an absolute one.

    `stored_path` is written by `app/services/dataset_service.py` as
    `str(storage_dir / filename)`, where `storage_dir` defaults to the
    relative path `"storage/datasets"` -- relative to whatever directory
    the FastAPI app was running from (`backend/`) at upload time. Resolve
    it against `_BACKEND_DIR`, not the current process's cwd, since the
    MCP server can be launched from anywhere (see backend/mcp/README.md's
    naming-collision notes on why cwd shouldn't be relied on here).
    """
    path = record.stored_path
    if not path.is_absolute():
        path = _BACKEND_DIR / path
    return path


def load_dataframe(dataset_id: str) -> tuple[pd.DataFrame, DatasetRecord]:
    """Resolve `dataset_id` to its row, then load its file into a DataFrame.

    Only `.csv` is supported today -- matching
    `app/services/dataset_service.py`'s `_inspect_csv`, the only file type
    this codebase's upload flow actually parses. Returns the record
    alongside the frame so callers (the MCP tools) have `name`/
    `target_column` without a second lookup.
    """
    record = get_dataset_record(dataset_id)
    path = resolve_file_path(record)

    if not path.exists():
        raise DatasetFileMissingError(dataset_id, path)

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise UnsupportedDatasetFileError(path)

    df = pd.read_csv(path)

    if df.shape[1] == 0:
        raise EmptyDatasetError(dataset_id, "file has no columns")
    if df.shape[0] == 0:
        raise EmptyDatasetError(dataset_id, "file has no data rows")

    return df, record
