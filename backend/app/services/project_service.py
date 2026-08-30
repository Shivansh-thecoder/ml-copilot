"""Business-logic layer for projects.

Routers translate HTTP <-> Python; this translates that into calls against
the repository, and is where domain rules will live as they appear (e.g. a
future "archived projects can't be renamed"). Today those rules are
minimal — mostly delegation — but routers should never call the
repository directly, so the seam is in place before there's much to put
in it.
"""

from app.repositories.project_repository import ProjectRepository
from app.schemas.project import Project, ProjectCreate, ProjectUpdate


class ProjectNotFoundError(Exception):
    """Raised when an operation targets a project id that doesn't exist.
    A plain exception, not an HTTPException, so this layer stays usable
    without FastAPI in the import path — app/core/exceptions.py is what
    decides this becomes a 404."""

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        super().__init__(f"Project '{project_id}' not found")


class ProjectService:
    def __init__(self, repository: ProjectRepository) -> None:
        self._repository = repository

    def list_projects(self) -> list[Project]:
        return self._repository.list()

    def get_project(self, project_id: str) -> Project:
        project = self._repository.get(project_id)
        if project is None:
            raise ProjectNotFoundError(project_id)
        return project

    def create_project(self, data: ProjectCreate) -> Project:
        return self._repository.create(data)

    def update_project(self, project_id: str, data: ProjectUpdate) -> Project:
        updated = self._repository.update(project_id, data)
        if updated is None:
            raise ProjectNotFoundError(project_id)
        return updated

    def delete_project(self, project_id: str) -> None:
        deleted = self._repository.delete(project_id)
        if not deleted:
            raise ProjectNotFoundError(project_id)
