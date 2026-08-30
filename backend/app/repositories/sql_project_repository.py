"""Database-backed implementation of `ProjectRepository`.

Satisfies the exact same abstract contract defined in
app/repositories/project_repository.py — that contract was written
against an in-memory dict specifically so this swap wouldn't touch
`ProjectService` or any router. The repository owns a `Session` (one per
request; see app/core/database.py) rather than the module-level dict
`InMemoryProjectRepository` used, since a database session — unlike a
plain dict — can't be shared across concurrent requests.
"""

import uuid

from sqlmodel import Session, select

from app.models.project import ProjectTable
from app.repositories.project_repository import ProjectRepository
from app.schemas.project import Project, ProjectCreate, ProjectUpdate


class SqlProjectRepository(ProjectRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

    def list(self) -> list[Project]:
        rows = self._session.exec(select(ProjectTable)).all()
        return [Project.model_validate(row, from_attributes=True) for row in rows]

    def get(self, project_id: str) -> Project | None:
        row = self._session.get(ProjectTable, project_id)
        return Project.model_validate(row, from_attributes=True) if row else None

    def create(self, data: ProjectCreate) -> Project:
        row = ProjectTable(id=str(uuid.uuid4()), **data.model_dump())
        self._session.add(row)
        self._session.commit()
        self._session.refresh(row)
        return Project.model_validate(row, from_attributes=True)

    def update(self, project_id: str, data: ProjectUpdate) -> Project | None:
        row = self._session.get(ProjectTable, project_id)
        if row is None:
            return None
        # Same exclude_unset reasoning as InMemoryProjectRepository: only
        # fields the client actually sent get overwritten.
        changes = data.model_dump(exclude_unset=True)
        for field, value in changes.items():
            setattr(row, field, value)
        self._session.add(row)
        self._session.commit()
        self._session.refresh(row)
        return Project.model_validate(row, from_attributes=True)

    def delete(self, project_id: str) -> bool:
        row = self._session.get(ProjectTable, project_id)
        if row is None:
            return False
        self._session.delete(row)
        self._session.commit()
        return True
