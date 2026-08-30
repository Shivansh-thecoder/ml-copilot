"""Dependency providers wired up via FastAPI's `Depends()`.

Each request gets its own `Session` (see app/core/database.py) and a
fresh, session-backed repository built on top of it — session-per-request,
never a cached session shared across concurrent requests. The service
layer and routers depend only on these providers, so swapping the storage
backend is confined to this file.
"""

from fastapi import Depends
from sqlmodel import Session

from app.core.database import get_session
from app.repositories.dataset_repository import DatasetRepository
from app.repositories.project_repository import ProjectRepository
from app.repositories.sql_dataset_repository import SqlDatasetRepository
from app.repositories.sql_project_repository import SqlProjectRepository
from app.services.dataset_service import DatasetService
from app.services.project_service import ProjectService


def get_project_repository(session: Session = Depends(get_session)) -> ProjectRepository:
    return SqlProjectRepository(session)


def get_project_service(
    repository: ProjectRepository = Depends(get_project_repository),
) -> ProjectService:
    return ProjectService(repository)


def get_dataset_repository(session: Session = Depends(get_session)) -> DatasetRepository:
    return SqlDatasetRepository(session)


def get_dataset_service(
    repository: DatasetRepository = Depends(get_dataset_repository),
) -> DatasetService:
    return DatasetService(repository)
