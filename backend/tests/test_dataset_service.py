"""Service-layer tests for datasets. Uses the in-memory repository (fast,
no database) plus a real small CSV UploadFile — but redirects
`settings.storage_dir` to pytest's `tmp_path` first, so these tests write
into a throwaway directory instead of the app's real storage/ folder.
"""

import io

import pytest
from fastapi import UploadFile

from app.core.config import settings
from app.repositories.dataset_repository import InMemoryDatasetRepository
from app.schemas.dataset import DatasetCreate, DatasetUpdate
from app.services.dataset_service import DatasetNotFoundError, DatasetService

_CSV_CONTENT = b"a,b,c\n1,2,3\n4,5,6\n"


@pytest.fixture
def service(tmp_path, monkeypatch) -> DatasetService:
    monkeypatch.setattr(settings, "storage_dir", str(tmp_path))
    return DatasetService(InMemoryDatasetRepository())


def _csv_upload(filename: str = "sample.csv") -> UploadFile:
    return UploadFile(file=io.BytesIO(_CSV_CONTENT), filename=filename)


def test_create_dataset_inspects_csv_and_saves_file(service: DatasetService, tmp_path):
    created = service.create_dataset(DatasetCreate(name="sample", target_column="c"), _csv_upload())

    assert created.rows == 2  # 2 data rows, header excluded
    assert created.columns == 3
    assert created.size_bytes == len(_CSV_CONTENT)
    assert (tmp_path / f"{created.id}_sample.csv").exists()


def test_create_dataset_non_csv_gets_zero_rows_columns(service: DatasetService):
    upload = UploadFile(file=io.BytesIO(b"not-a-csv"), filename="notes.txt")
    created = service.create_dataset(DatasetCreate(name="notes"), upload)

    assert created.rows == 0
    assert created.columns == 0


def test_get_missing_raises(service: DatasetService):
    with pytest.raises(DatasetNotFoundError):
        service.get_dataset("does-not-exist")


def test_update_missing_raises(service: DatasetService):
    with pytest.raises(DatasetNotFoundError):
        service.update_dataset("does-not-exist", DatasetUpdate(name="x"))


def test_delete_removes_row_and_file(service: DatasetService, tmp_path):
    created = service.create_dataset(DatasetCreate(name="sample"), _csv_upload())
    stored_file = tmp_path / f"{created.id}_sample.csv"
    assert stored_file.exists()

    service.delete_dataset(created.id)

    assert not stored_file.exists()
    with pytest.raises(DatasetNotFoundError):
        service.get_dataset(created.id)


def test_delete_missing_raises(service: DatasetService):
    with pytest.raises(DatasetNotFoundError):
        service.delete_dataset("does-not-exist")
