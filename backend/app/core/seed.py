"""One-time dev-convenience seed data.

Inserts two sample projects so `GET /projects` isn't empty on a fresh
database — but only when the `projects` table is empty, so it's safe to
call unconditionally on every app startup without duplicating rows or
clobbering real data.
"""

from sqlmodel import Session, select

from app.models.project import ProjectTable

_SEED_PROJECTS: list[ProjectTable] = [
    ProjectTable(
        id="customer-churn",
        name="Customer Churn",
        description="Predict which customers are likely to cancel their subscription.",
        dataset_name="telco_customers.csv",
        last_opened="2 days ago",
        status="active",
    ),
    ProjectTable(
        id="fraud-detection",
        name="Fraud Detection",
        description="Flag suspicious transactions in real time.",
        dataset_name="transactions_2024.csv",
        last_opened="5 hours ago",
        status="active",
    ),
]


def seed_projects(session: Session) -> None:
    already_seeded = session.exec(select(ProjectTable)).first() is not None
    if already_seeded:
        return
    session.add_all(_SEED_PROJECTS)
    session.commit()
