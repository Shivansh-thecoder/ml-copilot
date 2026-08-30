"""Storage abstraction for datasets — same reasoning as
app/repositories/project_repository.py: an abstract contract plus an
in-memory implementation kept around for fast service-layer tests. The
running app is wired (via app/api/dependencies.py) to SqlDatasetRepository
instead.

`create()` takes a fully-formed `Dataset`, not a `DatasetCreate` (unlike
ProjectRepository.create, which takes ProjectCreate): a dataset's id has
to exist *before* its file is saved to disk — the stored filename is
derived from it — and rows/columns/size are only known after
DatasetService inspects the uploaded file. So the service assembles the
complete entity and the repository just persists it.
"""

from abc import ABC, abstractmethod

from app.schemas.dataset import Dataset, DatasetUpdate


class DatasetRepository(ABC):
    """Storage contract for datasets, independent of what backs it."""

    @abstractmethod
    def list(self) -> list[Dataset]: ...

    @abstractmethod
    def get(self, dataset_id: str) -> Dataset | None: ...

    @abstractmethod
    def create(self, dataset: Dataset) -> Dataset: ...

    @abstractmethod
    def update(self, dataset_id: str, data: DatasetUpdate) -> Dataset | None: ...

    @abstractmethod
    def delete(self, dataset_id: str) -> bool: ...


class InMemoryDatasetRepository(DatasetRepository):
    """Dict-backed storage, kept for service-layer tests only — the
    running app always uses SqlDatasetRepository."""

    def __init__(self) -> None:
        self._datasets: dict[str, Dataset] = {}

    def list(self) -> list[Dataset]:
        return list(self._datasets.values())

    def get(self, dataset_id: str) -> Dataset | None:
        return self._datasets.get(dataset_id)

    def create(self, dataset: Dataset) -> Dataset:
        self._datasets[dataset.id] = dataset
        return dataset

    def update(self, dataset_id: str, data: DatasetUpdate) -> Dataset | None:
        existing = self._datasets.get(dataset_id)
        if existing is None:
            return None
        changes = data.model_dump(exclude_unset=True)
        updated = existing.model_copy(update=changes)
        self._datasets[dataset_id] = updated
        return updated

    def delete(self, dataset_id: str) -> bool:
        return self._datasets.pop(dataset_id, None) is not None
