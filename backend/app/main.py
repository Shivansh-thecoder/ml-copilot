"""FastAPI application entrypoint.

Wires together the pieces the other modules define: settings, the
aggregated API router, domain-exception -> HTTP-response handlers, and a
startup seed step. Deliberately thin — anything more than wiring belongs
in api/, services/, or core/, not here.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Session

from app.api.router import api_router
from app.core.config import settings
from app.core.database import engine
from app.core.exceptions import register_exception_handlers
from app.core.seed import seed_projects


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once when the app starts, before it accepts requests. Seeding
    # here (rather than e.g. inside get_session) means it happens exactly
    # once per process, not on every request.
    with Session(engine) as session:
        seed_projects(session)
    yield


app = FastAPI(
    title=settings.app_name,
    description="Backend service for the ML Copilot workspace.",
    version=settings.app_version,
    lifespan=lifespan,
)

# Origins come from Settings (app/core/config.py) rather than a "*"
# wildcard: browsers reject `allow_credentials=True` combined with a
# wildcard origin outright, and an explicit list is what a deployed
# frontend origin will need to be added to later anyway.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)
register_exception_handlers(app)


@app.get("/health")
def health_check() -> dict[str, str]:
    """Basic liveness check used by Docker Compose / uptime checks."""
    return {"status": "ok"}
