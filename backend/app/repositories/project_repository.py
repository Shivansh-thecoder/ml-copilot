"""In-memory repository for Project entities.

The abstract base class exists even though there's only one implementation
today: it documents the storage contract the service layer relies on, so a
future `SqlAlchemyProjectRepository` can be dropped in without the service
or routers knowing storage changed at all.
"""

import uuid
from abc import ABC, abstractmethod

from app.schemas.project import Project, ProjectCreate, ProjectUpdate


class ProjectRepository(ABC):
    """Storage contract for projects, independent of what backs it."""

    @abstractmethod
    def list(self) -> list[Project]: ...

    @abstractmethod
    def get(self, project_id: str) -> Project | None: ...

    @abstractmethod
    def create(self, data: ProjectCreate) -> Project: ...

    @abstractmethod
    def update(self, project_id: str, data: ProjectUpdate) -> Project | None: ...

    @abstractmethod
    def delete(self, project_id: str) -> bool: ...


# Mirrors frontend/lib/sampleProjects.ts so the API returns recognizable
# data the moment the frontend is wired up to it, instead of an empty list.
# Purely a dev convenience — nothing depends on these specific rows existing.
_SEED_PROJECTS: list[Project] = [
    Project(
        id="customer-churn",
        name="Customer Churn",
        description="Predict which customers are likely to cancel their subscription.",
        dataset_name="telco_customers.csv",
        last_opened="2 days ago",
        status="active",
    ),
    Project(
        id="fraud-detection",
        name="Fraud Detection",
        description="Flag suspicious transactions in real time.",
        dataset_name="transactions_2024.csv",
        last_opened="5 hours ago",
        status="active",
    ),
]


class InMemoryProjectRepository(ProjectRepository):
    """Dict-backed storage, keyed by id. Not thread-safe and wiped on
    restart — acceptable for this phase, since the goal is to prove out the
    repository *seam* that a real database will later sit behind, not to
    persist anything durably yet."""

    def __init__(self) -> None:
        self._projects: dict[str, Project] = {p.id: p for p in _SEED_PROJECTS}

    def list(self) -> list[Project]:
        return list(self._projects.values())

    def get(self, project_id: str) -> Project | None:
        return self._projects.get(project_id)

    def create(self, data: ProjectCreate) -> Project:
        project = Project(id=str(uuid.uuid4()), **data.model_dump())
        self._projects[project.id] = project
        return project

    def update(self, project_id: str, data: ProjectUpdate) -> Project | None:
        existing = self._projects.get(project_id)
        if existing is None:
            return None
        # exclude_unset (not exclude_none) so a client explicitly clearing
        # `description` to "" is honored, while fields it didn't mention
        # at all are left untouched.
        changes = data.model_dump(exclude_unset=True)
        updated = existing.model_copy(update=changes)
        self._projects[project_id] = updated
        return updated

    def delete(self, project_id: str) -> bool:
        return self._projects.pop(project_id, None) is not None
