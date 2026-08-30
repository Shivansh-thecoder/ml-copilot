"""Aggregates every feature router into one router main.py can include.

New resource routers are registered here, so main.py stays a fixed few
lines regardless of how many resources exist.
"""

from fastapi import APIRouter

from app.api.routers import copilot, datasets, models, projects

api_router = APIRouter()
api_router.include_router(projects.router)
api_router.include_router(datasets.router)
api_router.include_router(copilot.router)
api_router.include_router(models.router)
