"""Shared pytest fixtures.

Two different kinds of double back these tests, deliberately: repository
tests (test_*_repository.py) use a throwaway *real* SQLite database (this
`session` fixture) because what's under test is the SQL mapping itself,
which a fake can't verify. Service tests (test_*_service.py) instead use
the in-memory repositories kept in app/repositories/ for exactly this
purpose — ProjectService/DatasetService's own logic is what's under test
there, not SQL, so a fast in-memory fake is the right tool.
"""

import pytest
from sqlmodel import Session, SQLModel, create_engine

# Importing app.models registers ProjectTable/DatasetTable on
# SQLModel.metadata — without it, create_all() below would create an
# empty database with no tables.
import app.models  # noqa: F401


@pytest.fixture
def session():
    # A fresh in-memory SQLite database per test, not the app's real
    # `app.db` — each test gets an isolated, empty schema and nothing
    # persists between tests or touches the real database file.
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
