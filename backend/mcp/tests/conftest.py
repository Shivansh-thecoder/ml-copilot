"""Shared fixtures for the ml_engine/MCP tool test suite.

`sys.path` gets `backend/mcp` (this file's parent) inserted -- the same
directory Python puts on `sys.path[0]` automatically when `server.py` is
run as a script -- so `import ml_engine`, `import tools` work the same
way here as they do for the real server. Deliberately `backend/mcp`, not
`backend`: see pytest.ini's comment and README.md's naming-collision
notes for why inserting `backend` itself would be a real bug, not just a
style choice.
"""

import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def binary_classification_df() -> pd.DataFrame:
    """Small, deterministic binary classification dataset -- two informative
    numeric features, a churn-like binary target, no missing values."""
    rng = np.random.default_rng(42)
    n = 200
    feature_1 = rng.normal(0, 1, n)
    feature_2 = rng.normal(0, 1, n)
    logit = 1.5 * feature_1 - 1.0 * feature_2
    probability = 1 / (1 + np.exp(-logit))
    target = (rng.uniform(0, 1, n) < probability).astype(int)
    return pd.DataFrame({"feature_1": feature_1, "feature_2": feature_2, "churned": target})


@pytest.fixture
def regression_df() -> pd.DataFrame:
    """Small, deterministic regression dataset -- linear-ish relationship plus noise."""
    rng = np.random.default_rng(7)
    n = 200
    feature_1 = rng.normal(0, 1, n)
    feature_2 = rng.normal(0, 1, n)
    noise = rng.normal(0, 0.5, n)
    target = 3.0 * feature_1 - 2.0 * feature_2 + 5.0 + noise
    return pd.DataFrame({"feature_1": feature_1, "feature_2": feature_2, "price": target})


@pytest.fixture
def mixed_types_df() -> pd.DataFrame:
    """Numeric + categorical features together."""
    rng = np.random.default_rng(11)
    n = 150
    numeric_feature = rng.normal(0, 1, n)
    category = rng.choice(["A", "B", "C"], size=n)
    bump = np.where(category == "A", 1.0, np.where(category == "B", -1.0, 0.0))
    target = (numeric_feature + bump + rng.normal(0, 0.5, n) > 0).astype(int)
    return pd.DataFrame({"numeric_feature": numeric_feature, "category": category, "label": target})


@pytest.fixture
def missing_values_df() -> pd.DataFrame:
    """Same shape as mixed_types_df, but with deliberate missing values in
    both a numeric and a categorical column."""
    rng = np.random.default_rng(13)
    n = 100
    numeric_feature = rng.normal(0, 1, n)
    category = rng.choice(["A", "B", "C"], size=n).astype(object)
    target = (numeric_feature + rng.normal(0, 0.5, n) > 0).astype(int)

    numeric_feature = numeric_feature.astype(object)
    missing_numeric_idx = rng.choice(n, size=10, replace=False)
    for i in missing_numeric_idx:
        numeric_feature[i] = None
    missing_cat_idx = rng.choice(n, size=8, replace=False)
    for i in missing_cat_idx:
        category[i] = None

    return pd.DataFrame({"numeric_feature": pd.to_numeric(pd.Series(numeric_feature)), "category": category, "label": target})


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """Binary classification with severe (95/5) class imbalance."""
    rng = np.random.default_rng(21)
    n = 300
    feature_1 = rng.normal(0, 1, n)
    feature_2 = rng.normal(0, 1, n)
    target = np.zeros(n, dtype=int)
    minority_idx = rng.choice(n, size=15, replace=False)
    target[minority_idx] = 1
    return pd.DataFrame({"feature_1": feature_1, "feature_2": feature_2, "fraud": target})


@pytest.fixture
def ambiguous_target_df() -> pd.DataFrame:
    """No column looks like an obvious target -- all generically named,
    similar cardinality, no name hints."""
    rng = np.random.default_rng(5)
    n = 60
    return pd.DataFrame(
        {
            "col_a": rng.normal(0, 1, n),
            "col_b": rng.normal(0, 1, n),
            "col_c": rng.normal(0, 1, n),
        }
    )


@pytest.fixture
def seeded_dataset(tmp_path) -> dict:
    """Write a CSV + a matching `datasets` row into a throwaway sqlite db,
    monkeypatching `ml_engine.dataset_io` to point at them -- exercises the
    real dataset_id -> DataFrame resolution path without touching the
    real backend/app.db or backend/storage/.

    Returns {"dataset_id": ..., "df": ..., "csv_path": ..., "db_path": ...}.
    """
    import ml_engine.dataset_io as dataset_io

    rng = np.random.default_rng(99)
    n = 40
    age = rng.integers(18, 70, n)
    income = rng.integers(30000, 90000, n)
    plan = rng.choice(["basic", "pro"], size=n)
    logit = -0.03 * (age - 40) + np.where(plan == "basic", 0.8, -0.8)
    probability = 1 / (1 + np.exp(-logit))
    churned = (rng.uniform(0, 1, n) < probability).astype(int)
    df = pd.DataFrame({"age": age, "income": income, "plan": plan, "churned": churned})
    csv_path = tmp_path / "seeded.csv"
    df.to_csv(csv_path, index=False)

    db_path = tmp_path / "test_app.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        "CREATE TABLE datasets (id TEXT PRIMARY KEY, name TEXT, target_column TEXT, stored_path TEXT)"
    )
    conn.execute(
        "INSERT INTO datasets (id, name, target_column, stored_path) VALUES (?, ?, ?, ?)",
        ("ds-001", "seeded", "churned", str(csv_path)),
    )
    conn.commit()
    conn.close()

    original_db_path = dataset_io._DB_PATH
    dataset_io._DB_PATH = db_path
    try:
        yield {"dataset_id": "ds-001", "df": df, "csv_path": csv_path, "db_path": db_path}
    finally:
        dataset_io._DB_PATH = original_db_path


@pytest.fixture(autouse=True)
def isolated_ml_storage(tmp_path, monkeypatch):
    """Redirect ml_engine.persistence's storage dirs to a per-test tmp_path
    so tests never write into the real backend/mcp/storage/."""
    import ml_engine.persistence as persistence

    monkeypatch.setattr(persistence, "_STORAGE_DIR", tmp_path / "storage")
    monkeypatch.setattr(persistence, "_MODELS_DIR", tmp_path / "storage" / "models")
    monkeypatch.setattr(persistence, "_SAVED_DIR", tmp_path / "storage" / "saved")
    monkeypatch.setattr(persistence, "_EXPERIMENTS_DB", tmp_path / "storage" / "experiments.db")
