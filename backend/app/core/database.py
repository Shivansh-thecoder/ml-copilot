"""Database engine and session management.

SQLModel wraps SQLAlchemy Core + Pydantic: a `SQLModel` subclass with
`table=True` (see app/models/) doubles as both an ORM table definition and
a validated Python object, so it needs only one shared engine/session
setup, defined here once and reused by every repository.

Session-per-request (`get_session` below) rather than one global session:
each request gets its own SQLAlchemy `Session`, which is what makes
concurrent requests safe — sharing a single session across requests would
let one request's uncommitted changes leak into another's queries.
"""

from sqlmodel import Session, SQLModel, create_engine

from app.core.config import settings

# `check_same_thread` only matters for SQLite — it forbids using a
# connection outside the thread that created it by default, which breaks
# under FastAPI's threaded request handling. Every other database
# SQLAlchemy supports ignores this argument, so it's safe to pass
# unconditionally... except create_engine() rejects unknown connect_args
# for some drivers, hence the guard.
connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}

engine = create_engine(settings.database_url, connect_args=connect_args)


def create_db_and_tables() -> None:
    """Creates every table registered on `SQLModel.metadata`.

    Not used by the running app or by Alembic — Alembic's migration
    scripts (alembic/versions/) are the source of truth for schema in any
    environment that matters. This exists only for the test suite (see
    tests/conftest.py), which wants a fresh throwaway schema per test run
    without invoking Alembic each time.
    """
    SQLModel.metadata.create_all(engine)


def get_session():
    """FastAPI dependency: yields one Session per request via `Depends()`,
    and closes it once the response is sent (the `with` block). This is
    the same session-per-request lifecycle virtually every SQLAlchemy web
    app uses, and is why app/api/dependencies.py builds a fresh
    SqlProjectRepository/SqlDatasetRepository per request rather than
    caching one.
    """
    with Session(engine) as session:
        yield session
