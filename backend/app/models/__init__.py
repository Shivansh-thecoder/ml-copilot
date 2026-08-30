# SQLModel table entities — the database-mapped counterpart to
# app/schemas/'s request/response models. Importing the submodules here
# (rather than leaving callers to import them individually) guarantees
# both tables are registered on SQLModel.metadata as soon as `app.models`
# is imported anywhere — required by both Alembic's autogenerate (see
# alembic/env.py) and tests/conftest.py's create_all().
from app.models.dataset import DatasetTable
from app.models.project import ProjectTable

__all__ = ["DatasetTable", "ProjectTable"]
