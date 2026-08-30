"""Database-backed implementation of `DatasetRepository`. Same pattern as
SqlProjectRepository (see app/repositories/sql_project_repository.py):
satisfies the abstract contract in dataset_repository.py, owns a
request-scoped `Session` rather than an in-process dict.
"""

from sqlmodel import Session, select

from app.models.dataset import DatasetTable
from app.repositories.dataset_repository import DatasetRepository
from app.schemas.dataset import Dataset, DatasetUpdate


class SqlDatasetRepository(DatasetRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

    def list(self) -> list[Dataset]:
        rows = self._session.exec(select(DatasetTable)).all()
        return [Dataset.model_validate(row, from_attributes=True) for row in rows]

    def get(self, dataset_id: str) -> Dataset | None:
        row = self._session.get(DatasetTable, dataset_id)
        return Dataset.model_validate(row, from_attributes=True) if row else None

    def create(self, dataset: Dataset) -> Dataset:
        row = DatasetTable(**dataset.model_dump())
        self._session.add(row)
        self._session.commit()
        self._session.refresh(row)
        return Dataset.model_validate(row, from_attributes=True)

    def update(self, dataset_id: str, data: DatasetUpdate) -> Dataset | None:
        row = self._session.get(DatasetTable, dataset_id)
        if row is None:
            return None
        changes = data.model_dump(exclude_unset=True)
        for field, value in changes.items():
            setattr(row, field, value)
        self._session.add(row)
        self._session.commit()
        self._session.refresh(row)
        return Dataset.model_validate(row, from_attributes=True)

    def delete(self, dataset_id: str) -> bool:
        row = self._session.get(DatasetTable, dataset_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.commit()
        return True
