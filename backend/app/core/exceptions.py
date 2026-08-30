"""Maps domain exceptions raised by the service layer to HTTP responses.

Services and repositories raise plain Python exceptions (e.g.
NotFoundError) rather than HTTPException — that keeps them usable outside
of an HTTP context (a future CLI, background job, or test) without
importing FastAPI. Registering one handler per exception type here means
every router gets consistent status codes/response bodies for free,
instead of each route handler writing its own try/except.
"""

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.services.dataset_service import DatasetNotFoundError
from app.services.project_service import ProjectNotFoundError


async def _not_found_handler(
    request: Request, exc: ProjectNotFoundError | DatasetNotFoundError
) -> JSONResponse:
    return JSONResponse(status_code=404, content={"detail": str(exc)})


def register_exception_handlers(app: FastAPI) -> None:
    """Called once from main.py at startup. Both "not found" errors reuse
    the same handler body — they only differ in which exception type
    triggers it — so each resource just registers its own exception
    against the shared handler rather than writing its own copy."""
    app.add_exception_handler(ProjectNotFoundError, _not_found_handler)
    app.add_exception_handler(DatasetNotFoundError, _not_found_handler)
