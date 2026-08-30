"""Service-layer tests use the in-memory repository, not a database:
ProjectService's own logic (raising ProjectNotFoundError on a missing id)
is what's under test here, not SQL — a fast fake repository is the right
tool, and it's what keeps these tests independent of test_project_
repository.py's database fixture.
"""

import pytest

from app.repositories.project_repository import InMemoryProjectRepository
from app.schemas.project import ProjectCreate, ProjectStatus, ProjectUpdate
from app.services.project_service import ProjectNotFoundError, ProjectService


@pytest.fixture
def service() -> ProjectService:
    return ProjectService(InMemoryProjectRepository())


def test_create_then_get(service: ProjectService):
    created = service.create_project(ProjectCreate(name="Test"))
    assert service.get_project(created.id).name == "Test"


def test_get_missing_raises(service: ProjectService):
    with pytest.raises(ProjectNotFoundError):
        service.get_project("does-not-exist")


def test_update_applies_partial_changes(service: ProjectService):
    created = service.create_project(ProjectCreate(name="Test", description="original"))
    updated = service.update_project(created.id, ProjectUpdate(status=ProjectStatus.ARCHIVED))
    assert updated.status == ProjectStatus.ARCHIVED
    assert updated.description == "original"


def test_update_missing_raises(service: ProjectService):
    with pytest.raises(ProjectNotFoundError):
        service.update_project("does-not-exist", ProjectUpdate(name="x"))


def test_delete_then_get_raises(service: ProjectService):
    created = service.create_project(ProjectCreate(name="Test"))
    service.delete_project(created.id)
    with pytest.raises(ProjectNotFoundError):
        service.get_project(created.id)


def test_delete_missing_raises(service: ProjectService):
    with pytest.raises(ProjectNotFoundError):
        service.delete_project("does-not-exist")
