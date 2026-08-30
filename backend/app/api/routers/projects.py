"""HTTP layer for the Project resource.

Route handlers only translate between HTTP and the service layer — parse
the path/body, call ProjectService, return the result. No handler touches
the repository or holds any business logic; that separation is the point
of having a service layer at all.
"""

from fastapi import APIRouter, Depends, status

from app.api.dependencies import get_project_service
from app.schemas.project import ProjectCreate, ProjectResponse, ProjectUpdate
from app.services.project_service import ProjectService

router = APIRouter(prefix="/projects", tags=["projects"])


@router.get("", response_model=list[ProjectResponse])
def list_projects(
    service: ProjectService = Depends(get_project_service),
) -> list[ProjectResponse]:
    return service.list_projects()


@router.get("/{project_id}", response_model=ProjectResponse)
def get_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    # ProjectNotFoundError propagates up to the handler registered in
    # app/core/exceptions.py, which turns it into a 404 — no try/except
    # needed here.
    return service.get_project(project_id)


@router.post("", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
def create_project(
    data: ProjectCreate,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    return service.create_project(data)


@router.put("/{project_id}", response_model=ProjectResponse)
def update_project(
    project_id: str,
    data: ProjectUpdate,
    service: ProjectService = Depends(get_project_service),
) -> ProjectResponse:
    return service.update_project(project_id, data)


@router.delete("/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_project(
    project_id: str,
    service: ProjectService = Depends(get_project_service),
) -> None:
    service.delete_project(project_id)
