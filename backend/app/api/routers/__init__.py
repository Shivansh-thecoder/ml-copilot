# One module per resource (projects.py, and later datasets.py,
# models.py, ...). Each defines its own APIRouter with its own prefix/tags;
# app/api/router.py collects them into the single router main.py includes.
