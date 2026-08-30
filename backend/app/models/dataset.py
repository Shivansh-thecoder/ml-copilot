"""SQLModel table definition for Dataset. See app/models/project.py for
why this is kept separate from app/schemas/dataset.py's Pydantic models.

Stores `stored_path` (where the uploaded file lives under storage/) and
`size_bytes` (the raw byte count) — internal details `DatasetResponse`
deliberately doesn't expose to the API; see app/schemas/dataset.py.
"""

from datetime import datetime

from sqlmodel import Field, SQLModel


class DatasetTable(SQLModel, table=True):
    __tablename__ = "datasets"

    id: str = Field(primary_key=True)
    name: str
    target_column: str = ""
    rows: int = 0
    columns: int = 0
    size_bytes: int = 0
    stored_path: str
    uploaded_at: datetime
