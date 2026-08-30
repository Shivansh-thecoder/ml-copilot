"""Repository-layer tests: exercise SqlProjectRepository against a real
(throwaway) SQLite database — the point of this layer is the SQL mapping
itself (does create() actually insert a row, does update() only touch the
fields it was given), which a fake repository can't verify."""

from app.repositories.sql_project_repository import SqlProjectRepository
from app.schemas.project import ProjectCreate, ProjectStatus, ProjectUpdate


def test_create_assigns_id_and_persists(session):
    repo = SqlProjectRepository(session)

    created = repo.create(ProjectCreate(name="Test Project", description="desc"))

    assert created.id
    assert created.status == ProjectStatus.ACTIVE  # server default applied
    fetched = repo.get(created.id)
    assert fetched is not None
    assert fetched.name == "Test Project"


def test_list_returns_every_row(session):
    repo = SqlProjectRepository(session)
    repo.create(ProjectCreate(name="A"))
    repo.create(ProjectCreate(name="B"))

    assert {p.name for p in repo.list()} == {"A", "B"}


def test_get_missing_returns_none(session):
    repo = SqlProjectRepository(session)
    assert repo.get("does-not-exist") is None


def test_update_only_changes_sent_fields(session):
    repo = SqlProjectRepository(session)
    created = repo.create(ProjectCreate(name="A", description="original"))

    updated = repo.update(created.id, ProjectUpdate(status=ProjectStatus.COMPLETED))

    assert updated is not None
    assert updated.status == ProjectStatus.COMPLETED
    assert updated.description == "original"  # untouched
    assert updated.name == "A"  # untouched


def test_update_missing_returns_none(session):
    repo = SqlProjectRepository(session)
    assert repo.update("does-not-exist", ProjectUpdate(name="x")) is None


def test_delete_removes_row_and_reports_result(session):
    repo = SqlProjectRepository(session)
    created = repo.create(ProjectCreate(name="A"))

    assert repo.delete(created.id) is True
    assert repo.get(created.id) is None
    assert repo.delete(created.id) is False  # already gone
