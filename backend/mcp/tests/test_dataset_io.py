import pandas as pd
import pytest

from ml_engine.dataset_io import (
    DatasetFileMissingError,
    DatasetNotFoundError,
    EmptyDatasetError,
    UnsupportedDatasetFileError,
    get_dataset_record,
    load_dataframe,
)


def test_load_dataframe_success(seeded_dataset):
    df, record = load_dataframe(seeded_dataset["dataset_id"])
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["age", "income", "plan", "churned"]
    assert record.id == "ds-001"
    assert record.target_column == "churned"


def test_get_dataset_record(seeded_dataset):
    record = get_dataset_record(seeded_dataset["dataset_id"])
    assert record.name == "seeded"


def test_nonexistent_dataset_id_raises(seeded_dataset):
    with pytest.raises(DatasetNotFoundError):
        get_dataset_record("does-not-exist")


def test_missing_db_file_raises_not_found(tmp_path, monkeypatch):
    import ml_engine.dataset_io as dataset_io

    monkeypatch.setattr(dataset_io, "_DB_PATH", tmp_path / "nonexistent.db")
    with pytest.raises(DatasetNotFoundError):
        get_dataset_record("anything")


def test_file_missing_on_disk_raises(seeded_dataset):
    seeded_dataset["csv_path"].unlink()
    with pytest.raises(DatasetFileMissingError):
        load_dataframe(seeded_dataset["dataset_id"])


def test_unsupported_file_type_raises(tmp_path, monkeypatch):
    import ml_engine.dataset_io as dataset_io
    import sqlite3

    bad_file = tmp_path / "data.json"
    bad_file.write_text("{}")

    db_path = tmp_path / "app.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE datasets (id TEXT PRIMARY KEY, name TEXT, target_column TEXT, stored_path TEXT)")
    conn.execute(
        "INSERT INTO datasets VALUES (?, ?, ?, ?)", ("ds-json", "bad", "", str(bad_file))
    )
    conn.commit()
    conn.close()

    monkeypatch.setattr(dataset_io, "_DB_PATH", db_path)
    with pytest.raises(UnsupportedDatasetFileError):
        load_dataframe("ds-json")


def test_empty_csv_raises(tmp_path, monkeypatch):
    import ml_engine.dataset_io as dataset_io
    import sqlite3

    empty_file = tmp_path / "empty.csv"
    empty_file.write_text("col_a,col_b\n")  # header only, zero data rows

    db_path = tmp_path / "app.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE datasets (id TEXT PRIMARY KEY, name TEXT, target_column TEXT, stored_path TEXT)")
    conn.execute("INSERT INTO datasets VALUES (?, ?, ?, ?)", ("ds-empty", "empty", "", str(empty_file)))
    conn.commit()
    conn.close()

    monkeypatch.setattr(dataset_io, "_DB_PATH", db_path)
    with pytest.raises(EmptyDatasetError):
        load_dataframe("ds-empty")
