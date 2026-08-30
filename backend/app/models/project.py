"""SQLModel table definition for Project.

Deliberately separate from `app/schemas/project.py`'s plain Pydantic
models even though SQLModel could let one class serve both roles: this is
what SQLAlchemy maps to an actual database row (`table=True`), while
schemas/ define the API's request/response contract. A column this table
needs later (e.g. a foreign key to a future `users` table) shouldn't
silently become an API field just because the classes were merged.
"""

from sqlalchemy import Column
from sqlalchemy import Enum as SAEnum
from sqlmodel import Field, SQLModel

from app.schemas.project import ProjectStatus

# SQLAlchemy's Enum column stores a Python Enum's *member name* ("ACTIVE")
# by default, not its `.value` ("active") — values_callable overrides that
# to persist `.value` instead, so a row inspected directly in the database
# reads "active", matching what the API and frontend both use, rather than
# a second, uppercase vocabulary that only exists inside this column.
_project_status_column = Column(
    SAEnum(ProjectStatus, values_callable=lambda enum_cls: [member.value for member in enum_cls]),
    nullable=False,
)


class ProjectTable(SQLModel, table=True):
    __tablename__ = "projects"

    id: str = Field(primary_key=True)
    name: str
    description: str = ""
    dataset_name: str = ""
    status: ProjectStatus = Field(default=ProjectStatus.ACTIVE, sa_column=_project_status_column)
    last_opened: str = "Just now"
