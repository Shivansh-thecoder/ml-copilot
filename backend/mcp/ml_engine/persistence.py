"""Model artifact persistence + a lightweight experiment log.

STORAGE LAYOUT
------------------
    backend/mcp/storage/
        experiments.db                        one sqlite file, one `experiments` table
        models/{run_id}__{model_name}.joblib   every candidate a training run produced
        saved/{dataset_id}.joblib + .json      the one model `save_model` designated "the" saved model for a dataset

WHY A SEPARATE `experiments.db`, NOT `backend/app.db`
------------------------------------------------------------
`app.db`'s schema is owned by Alembic (backend/alembic/versions/) --
adding a table via raw `CREATE TABLE IF NOT EXISTS`, bypassing Alembic
entirely, would leave that table invisible to future migrations, exactly
the schema drift Alembic exists to prevent. A second, purpose-built
SQLite file reuses "the existing backend/database architecture" in
spirit (SQLite, a plain table, primary-key lookups -- no new database
engine) without touching a schema something else owns. Not a full
MLflow replacement -- one table, no server, no UI.

WHY THE SAVED ARTIFACT IS A FULL PIPELINE, NOT JUST THE ESTIMATOR
------------------------------------------------------------------------
Every artifact saved here is a FITTED `sklearn.pipeline.Pipeline` --
preprocessing (imputation, scaling, encoding) and the estimator, as one
object (built in training.py/tuning.py). Loading it and calling
`.predict(raw_new_data)` re-applies the exact preprocessing the model was
trained against; saving only the estimator would silently assume whoever
loads it re-implements preprocessing identically -- exactly the kind of
divergence that breaks a model when it's actually used later.
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import joblib

_STORAGE_DIR = Path(__file__).resolve().parents[1] / "storage"
_MODELS_DIR = _STORAGE_DIR / "models"
_SAVED_DIR = _STORAGE_DIR / "saved"
_EXPERIMENTS_DB = _STORAGE_DIR / "experiments.db"


def _ensure_dirs() -> None:
    _MODELS_DIR.mkdir(parents=True, exist_ok=True)
    _SAVED_DIR.mkdir(parents=True, exist_ok=True)


def _connect() -> sqlite3.Connection:
    _ensure_dirs()
    conn = sqlite3.connect(str(_EXPERIMENTS_DB))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS experiments (
            run_id TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            dataset_name TEXT NOT NULL,
            model_name TEXT NOT NULL,
            task_type TEXT NOT NULL,
            target_column TEXT NOT NULL,
            primary_metric TEXT NOT NULL,
            cv_mean_json TEXT NOT NULL,
            cv_std_json TEXT NOT NULL,
            test_metrics_json TEXT NOT NULL,
            diagnostic TEXT NOT NULL,
            params_json TEXT NOT NULL,
            artifact_path TEXT NOT NULL,
            status TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (run_id, model_name)
        )
        """
    )
    return conn


def save_model_artifact(pipeline, run_id: str, model_name: str) -> Path:
    """Joblib-dump a fitted Pipeline to storage/models/, return its path."""
    _ensure_dirs()
    path = _MODELS_DIR / f"{run_id}__{model_name}.joblib"
    joblib.dump(pipeline, path)
    return path


def load_model_artifact(path: Path):
    """Load a previously saved fitted Pipeline. Raises FileNotFoundError if missing."""
    if not Path(path).exists():
        raise FileNotFoundError(f"No saved model artifact at {path}")
    return joblib.load(path)


def record_experiment(
    run_id: str,
    dataset_id: str,
    dataset_name: str,
    model_name: str,
    task_type: str,
    target_column: str,
    primary_metric: str,
    cv_mean: dict,
    cv_std: dict,
    test_metrics: dict,
    diagnostic: str,
    params: dict,
    artifact_path: Path,
    status: str = "completed",
) -> None:
    """Insert (or replace) one row: one candidate model from one training run."""
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT OR REPLACE INTO experiments
                (run_id, dataset_id, dataset_name, model_name, task_type, target_column,
                 primary_metric, cv_mean_json, cv_std_json, test_metrics_json, diagnostic,
                 params_json, artifact_path, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                dataset_id,
                dataset_name,
                model_name,
                task_type,
                target_column,
                primary_metric,
                json.dumps(cv_mean),
                json.dumps(cv_std),
                json.dumps(test_metrics),
                diagnostic,
                json.dumps(params),
                str(artifact_path),
                status,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_experiment(run_id: str, model_name: str | None = None) -> list[dict]:
    """Return experiment rows for a run_id (all candidates, or one model)."""
    conn = _connect()
    try:
        if model_name:
            cursor = conn.execute(
                "SELECT * FROM experiments WHERE run_id = ? AND model_name = ?", (run_id, model_name)
            )
        else:
            cursor = conn.execute("SELECT * FROM experiments WHERE run_id = ?", (run_id,))
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()
    return [_deserialize_row(row) for row in rows]


def list_experiments(dataset_id: str) -> list[dict]:
    """All experiment rows for a dataset, most recent first."""
    conn = _connect()
    try:
        cursor = conn.execute(
            "SELECT * FROM experiments WHERE dataset_id = ? ORDER BY created_at DESC", (dataset_id,)
        )
        columns = [d[0] for d in cursor.description]
        rows = [dict(zip(columns, row)) for row in cursor.fetchall()]
    finally:
        conn.close()
    return [_deserialize_row(row) for row in rows]


def _deserialize_row(row: dict) -> dict:
    for key in ("cv_mean_json", "cv_std_json", "test_metrics_json", "params_json"):
        new_key = key.replace("_json", "")
        row[new_key] = json.loads(row.pop(key))
    return row


def save_best_model(run_id: str, model_name: str, dataset_id: str) -> dict:
    """Copy one run's candidate artifact into storage/saved/ as THE saved
    model for this dataset.

    Overwrites any previously saved model for the same dataset_id -- one
    "current saved model" per dataset, not a history of every save (the
    `experiments` table already keeps the full history of runs this
    could be re-saved from).
    """
    rows = get_experiment(run_id, model_name)
    if not rows:
        raise ValueError(f"No experiment found for run_id={run_id!r}, model_name={model_name!r}")
    row = rows[0]

    pipeline = load_model_artifact(Path(row["artifact_path"]))

    _ensure_dirs()
    dest_path = _SAVED_DIR / f"{dataset_id}.joblib"
    joblib.dump(pipeline, dest_path)

    metadata = {
        "run_id": run_id,
        "model_name": model_name,
        "dataset_id": dataset_id,
        "task_type": row["task_type"],
        "target_column": row["target_column"],
        "primary_metric": row["primary_metric"],
        "cv_mean": row["cv_mean"],
        "test_metrics": row["test_metrics"],
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    meta_path = _SAVED_DIR / f"{dataset_id}.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    return {"saved_model_path": str(dest_path), "metadata": metadata}


def get_saved_model_metadata(dataset_id: str) -> dict | None:
    """Metadata for the current saved model for a dataset, or None if none saved yet."""
    meta_path = _SAVED_DIR / f"{dataset_id}.json"
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def load_saved_model(dataset_id: str):
    """Load the fitted Pipeline `save_best_model` designated for this dataset.

    Returns the same object training produced -- preprocessing and
    estimator together -- so callers predict on RAW rows and get the
    training-time preprocessing applied for free (see this module's
    docstring).
    """
    return load_model_artifact(_SAVED_DIR / f"{dataset_id}.joblib")


def list_saved_models() -> list[dict]:
    """Every dataset's currently saved model, most recently saved first.

    Reads the sidecar .json files rather than the `experiments` table:
    that table holds every candidate from every run, while `saved/` holds
    exactly the one model per dataset that was designated for use.
    """
    if not _SAVED_DIR.exists():
        return []
    models = [json.loads(path.read_text()) for path in _SAVED_DIR.glob("*.json")]
    return sorted(models, key=lambda m: m.get("saved_at", ""), reverse=True)
