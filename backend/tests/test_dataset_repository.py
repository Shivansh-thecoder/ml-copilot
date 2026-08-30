"""Repository-layer tests for datasets — same rationale as
test_project_repository.py: exercise SqlDatasetRepository against a real
database. `create()` takes a fully-formed Dataset here (unlike Project's
ProjectCreate), matching DatasetRepository's contract — see
app/repositories/dataset_repository.py for why."""

from datetime import datetime, timezone

from app.repositories.sql_dataset_repository import SqlDatasetRepository
from app.schemas.dataset import Dataset, DatasetUpdate


def _sample_dataset(dataset_id: str = "ds-1") -> Dataset:
    return Dataset(
        id=dataset_id,
        name="sample.csv",
        target_column="label",
        rows=10,
        columns=3,
        size_bytes=1024,
        stored_path=f"storage/datasets/{dataset_id}_sample.csv",
        uploaded_at=datetime.now(timezone.utc),
    )


def test_create_and_get(session):
    repo = SqlDatasetRepository(session)

    created = repo.create(_sample_dataset())

    fetched = repo.get(created.id)
    assert fetched is not None
    assert fetched.name == "sample.csv"
    assert fetched.rows == 10


def test_list_returns_every_row(session):
    repo = SqlDatasetRepository(session)
    repo.create(_sample_dataset("ds-1"))
    repo.create(_sample_dataset("ds-2"))

    assert {d.id for d in repo.list()} == {"ds-1", "ds-2"}


def test_update_only_changes_sent_fields(session):
    repo = SqlDatasetRepository(session)
    repo.create(_sample_dataset())

    updated = repo.update("ds-1", DatasetUpdate(name="renamed.csv"))

    assert updated is not None
    assert updated.name == "renamed.csv"
    assert updated.target_column == "label"  # untouched


def test_update_missing_returns_none(session):
    repo = SqlDatasetRepository(session)
    assert repo.update("does-not-exist", DatasetUpdate(name="x")) is None


def test_delete_removes_row(session):
    repo = SqlDatasetRepository(session)
    repo.create(_sample_dataset())

    assert repo.delete("ds-1") is True
    assert repo.get("ds-1") is None
    assert repo.delete("ds-1") is False
