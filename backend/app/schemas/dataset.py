"""Pydantic schemas for the Dataset resource.

Datasets differ structurally from Projects in one way: creation is a
multipart file upload, not a JSON body. `DatasetCreate` therefore only
describes the non-file form fields — the file itself is a separate
`UploadFile` parameter on the route (see app/api/routers/datasets.py), not
part of this schema.
"""

from datetime import datetime

from pydantic import BaseModel


class DatasetBase(BaseModel):
    name: str
    target_column: str = ""


class DatasetCreate(DatasetBase):
    """Form fields accompanying a file upload. No id/rows/columns/size/
    uploaded_at — DatasetService derives all of those only after the file
    is saved and inspected, so a client can't set them."""


class DatasetUpdate(BaseModel):
    """All optional, same reasoning as ProjectUpdate: renaming a dataset
    shouldn't require resending its target column."""

    name: str | None = None
    target_column: str | None = None


class Dataset(DatasetBase):
    """The full domain entity as held by the repository. `stored_path` and
    `size_bytes` are internal storage details — deliberately absent from
    `DatasetResponse` below, which is what actually crosses the wire."""

    id: str
    rows: int = 0
    columns: int = 0
    size_bytes: int = 0
    stored_path: str
    uploaded_at: datetime


class DatasetResponse(DatasetBase):
    """What every endpoint returns to the client. `size_label` is a
    human-readable rendering of `size_bytes` (e.g. "1.8 MB") computed at
    response time by `to_response()` below, rather than stored — so the
    display format can change without a migration."""

    id: str
    rows: int
    columns: int
    size_label: str
    uploaded_at: datetime

    model_config = {"from_attributes": True}


def format_size_label(size_bytes: int) -> str:
    """Render a byte count as a human-readable label (e.g. "1.8 MB",
    "460 KB") for display in the frontend's dataset list."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB"):
        if size < 1024:
            return f"{int(size)} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def to_response(dataset: Dataset) -> DatasetResponse:
    """Central place that hides `stored_path`/`size_bytes` from the API
    and derives `size_label` — every route that returns a dataset goes
    through this instead of repeating the field mapping.
    """
    return DatasetResponse(
        id=dataset.id,
        name=dataset.name,
        target_column=dataset.target_column,
        rows=dataset.rows,
        columns=dataset.columns,
        size_label=format_size_label(dataset.size_bytes),
        uploaded_at=dataset.uploaded_at,
    )
