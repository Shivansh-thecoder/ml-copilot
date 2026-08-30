"""Business logic for datasets.

Unlike ProjectService, this also coordinates the filesystem (via
app/core/storage.py) alongside the repository: deciding that a dataset's
file and its database row are created and deleted together is a domain
rule, not something the repository (which only knows about rows) or a
route handler should own.
"""

import csv
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import UploadFile

from app.core import storage
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.dataset import Dataset, DatasetCreate, DatasetUpdate


class DatasetNotFoundError(Exception):
    """Raised when an operation targets a dataset id that doesn't exist.
    A plain exception, not an HTTPException — see
    app/services/project_service.py's ProjectNotFoundError for the same
    reasoning; app/core/exceptions.py maps this to a 404."""

    def __init__(self, dataset_id: str) -> None:
        self.dataset_id = dataset_id
        super().__init__(f"Dataset '{dataset_id}' not found")


class DatasetService:
    def __init__(self, repository: DatasetRepository) -> None:
        self._repository = repository

    def list_datasets(self) -> list[Dataset]:
        return self._repository.list()

    def get_dataset(self, dataset_id: str) -> Dataset:
        dataset = self._repository.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        return dataset

    def create_dataset(self, data: DatasetCreate, file: UploadFile) -> Dataset:
        # The id is generated here (not by the repository, unlike
        # Project) because it's needed up front: it's part of the stored
        # filename, and the file has to be saved before rows/columns can
        # be inspected.
        dataset_id = str(uuid.uuid4())
        stored_path, size_bytes = storage.save_upload(dataset_id, file)
        rows, columns = _inspect_csv(stored_path)

        dataset = Dataset(
            id=dataset_id,
            name=data.name,
            target_column=data.target_column,
            rows=rows,
            columns=columns,
            size_bytes=size_bytes,
            stored_path=str(stored_path),
            uploaded_at=datetime.now(timezone.utc),
        )
        return self._repository.create(dataset)

    def update_dataset(self, dataset_id: str, data: DatasetUpdate) -> Dataset:
        updated = self._repository.update(dataset_id, data)
        if updated is None:
            raise DatasetNotFoundError(dataset_id)
        return updated

    def delete_dataset(self, dataset_id: str) -> None:
        dataset = self._repository.get(dataset_id)
        if dataset is None:
            raise DatasetNotFoundError(dataset_id)
        # File first, then row: if the file delete fails, the row (and
        # its stored_path) is still around to retry against; the reverse
        # order would leave an orphaned file nothing references.
        storage.delete_file(Path(dataset.stored_path))
        self._repository.delete(dataset_id)


def _inspect_csv(path: Path) -> tuple[int, int]:
    """Counts data rows/columns for .csv uploads. Non-CSV files get 0/0
    rather than a parse error — column counting only makes sense for
    tabular files, and this endpoint accepts any file type."""
    if path.suffix.lower() != ".csv":
        return 0, 0

    with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0, 0
        columns = len(header)
        rows = sum(1 for _ in reader)
    return rows, columns
