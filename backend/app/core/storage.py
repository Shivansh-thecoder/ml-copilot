"""Local filesystem storage for uploaded dataset files.

A real deployment would likely use object storage (S3, GCS); keeping
uploads on local disk under `settings.storage_dir` for now mirrors how
`app/repositories/` abstracts the database — swapping this module for an
S3-backed one later changes DatasetService's storage calls, not its
callers (the router).
"""

from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings

# 1 MiB chunks: large enough to be efficient, small enough that a big
# upload doesn't have to be held in memory all at once.
_CHUNK_SIZE = 1024 * 1024


def save_upload(dataset_id: str, upload: UploadFile) -> tuple[Path, int]:
    """Streams `upload` to disk under storage_dir, returning its saved
    path and size in bytes. Filenames are prefixed with `dataset_id` so
    two uploads named the same thing (e.g. "data.csv" twice) don't
    collide — the id is generated before this is called specifically so
    it can be used as that prefix.
    """
    storage_dir = Path(settings.storage_dir)
    storage_dir.mkdir(parents=True, exist_ok=True)

    destination = storage_dir / f"{dataset_id}_{upload.filename or 'upload'}"
    size = 0
    with destination.open("wb") as out:
        while chunk := upload.file.read(_CHUNK_SIZE):
            size += len(chunk)
            out.write(chunk)
    return destination, size


def delete_file(path: Path) -> None:
    """`missing_ok=True`: a dataset row can outlive its file (e.g. if a
    previous delete partially failed), and re-deleting shouldn't crash the
    request over a file that's already gone."""
    path.unlink(missing_ok=True)
