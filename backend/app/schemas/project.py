"""Pydantic schemas for the Project resource.

One file per resource (not one file per schema) because Project,
ProjectCreate, ProjectUpdate, and ProjectResponse are all facets of the
same concept and change together — adding a field almost always means
touching all four, so keeping them side by side makes that surface area
visible in one diff.
"""

from enum import Enum

from pydantic import BaseModel, Field


class ProjectStatus(str, Enum):
    """Mirrors the frontend's `ProjectStatus` union (frontend/lib/types.ts)
    so the two ends of the app agree on what a project's status can be.
    Subclassing `str` means this serializes as the plain string value
    ("active") in JSON rather than an object, matching what the frontend
    already expects."""

    ACTIVE = "active"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ProjectBase(BaseModel):
    """Fields a client provides regardless of whether they're creating or
    describing a project. Both `ProjectCreate` and the full `Project`
    entity build on this so the validation rules (e.g. name length) are
    defined once."""

    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    dataset_name: str = ""


class ProjectCreate(ProjectBase):
    """Body for `POST /projects`. Deliberately has no `id`, `status`, or
    `last_opened` fields — those are server-assigned, so a client can't
    set them by supplying them in the request body."""


class ProjectUpdate(BaseModel):
    """Body for `PUT /projects/{id}`. Every field is optional (unlike
    ProjectBase/ProjectCreate) so a client updating only `status` isn't
    forced to resend name/description it isn't changing. The service
    layer applies only the fields the client actually sent."""

    name: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = None
    dataset_name: str | None = None
    status: ProjectStatus | None = None


class Project(ProjectBase):
    """The full domain entity as held by the repository. Kept distinct
    from `ProjectResponse` even though they look identical today: if a
    later phase adds a field the repository needs to track but the API
    shouldn't expose (e.g. an internal `owner_id`), only this model
    changes — `ProjectResponse` keeps whatever contract the frontend
    already depends on."""

    id: str
    status: ProjectStatus = ProjectStatus.ACTIVE
    last_opened: str = "Just now"


class ProjectResponse(ProjectBase):
    """What every endpoint returns to the client. `from_attributes` isn't
    needed for building this from a `Project` (both are Pydantic models),
    but is left enabled since it costs nothing and is exactly what a
    future SQLAlchemy-backed repository would need."""

    id: str
    status: ProjectStatus
    last_opened: str

    model_config = {"from_attributes": True}
