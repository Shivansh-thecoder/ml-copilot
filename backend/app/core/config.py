"""Application-wide settings.

A single `Settings` object — rather than scattered `os.getenv()` calls or
hardcoded values in main.py — gives every later phase (Docker, tests, a
real database) one place to override configuration from the environment.
Nothing reads from `.env` yet outside of this file, so adding a setting
here is the only step needed to make it configurable.

WHY `backend/.env` HOLDS SETTINGS FOR TWO DIFFERENT SUBSYSTEMS
--------------------------------------------------------------------
This class's fields are all `MLCOPILOT_`-prefixed (`env_prefix` below) —
this app's own configuration. `backend/.env` also holds `OPENAI_API_KEY`
(unprefixed), which `backend/langgraph/llm/model.py` reads directly via
`os.getenv()`, deliberately NOT through this class — see that file's
docstring for why `backend/langgraph/` stays import-independent of
`backend/app/`. One shared `.env` file, two independent owners, each
reading only the keys that are theirs: that separation is what
`env_prefix` + `extra="ignore"` (below) encodes, not a loosening of
validation. See `_check_for_prefix_typos()` for how typo-catching is
preserved despite `extra="ignore"`.
"""

import os
from pathlib import Path

from dotenv import dotenv_values
from pydantic_settings import BaseSettings, SettingsConfigDict

_BACKEND_DIR = Path(__file__).resolve().parents[2]
_ENV_FILE = _BACKEND_DIR / ".env"
_ENV_PREFIX = "MLCOPILOT_"


class Settings(BaseSettings):
    app_name: str = "ML Copilot API"
    app_version: str = "0.1.0"

    # The Next.js dev server's default origin. A list (not a single str)
    # because a deployed frontend will eventually need a second, non-local
    # origin added alongside this one.
    cors_origins: list[str] = ["http://localhost:3000"]

    # SQLite by default: zero setup for local dev/tests, and a real
    # Postgres deployment later is a one-line env var change (SQLModel's
    # engine is driven entirely by this URL string) — nothing in
    # app/repositories/ or app/services/ needs to know which one is active.
    database_url: str = "sqlite:///./app.db"

    # Where uploaded dataset files are written (app/core/storage.py).
    # Local disk today; swapping to object storage (S3/GCS) later means
    # changing storage.py, not this setting's callers.
    storage_dir: str = "storage/datasets"

    # `extra="ignore"`, not the pydantic-settings default of "forbid":
    # `env_prefix` already declares this class's domain as
    # `MLCOPILOT_*` — variables outside that prefix (OPENAI_API_KEY, and
    # anything a future subsystem adds) are simply none of this class's
    # business, not "unrecognized fields" of ITS domain. "forbid" doesn't
    # honor that boundary; it flags every unmatched key regardless of
    # prefix, which is exactly what broke the moment `backend/.env`
    # started being shared. `_check_for_prefix_typos()` below is what
    # keeps this from silently swallowing an actual typo in a
    # MLCOPILOT_-prefixed name, which "ignore" alone would do.
    model_config = SettingsConfigDict(env_file=".env", env_prefix=_ENV_PREFIX, extra="ignore")


def _check_for_prefix_typos() -> None:
    """Fail loudly on a typo WITHIN this app's own `MLCOPILOT_` namespace.

    `extra="ignore"` (above) is necessary so unrelated variables sharing
    `backend/.env` (`OPENAI_API_KEY`, today) don't break this class — but
    left on its own, "ignore" would ALSO silently accept a typo like
    `MLCOPILOT_DATABSE_URL`, which is exactly the kind of mistake
    `extra="forbid"` used to catch. This function restores that
    protection, scoped correctly: it checks every `MLCOPILOT_`-prefixed
    key actually present (in `backend/.env` or the process environment)
    against the set this class's fields actually expect, and raises a
    clear error naming the mismatch if one doesn't match -- while never
    even looking at non-`MLCOPILOT_` keys, which are out of scope by
    design.
    """
    expected = {f"{_ENV_PREFIX}{name.upper()}" for name in Settings.model_fields}

    candidates: dict[str, str] = {}
    if _ENV_FILE.exists():
        candidates.update({key: "backend/.env" for key in dotenv_values(_ENV_FILE) if key})
    candidates.update({key: "the environment" for key in os.environ})

    unknown = sorted(
        key for key in candidates if key.upper().startswith(_ENV_PREFIX) and key.upper() not in expected
    )
    if unknown:
        known = ", ".join(sorted(expected))
        bad = ", ".join(f"'{key}' (from {candidates[key]})" for key in unknown)
        raise ValueError(
            f"Unrecognized {_ENV_PREFIX}* setting(s): {bad}. Known settings: {known}. "
            f"This usually means a typo -- fix the variable name, or add a new field to "
            f"Settings (app/core/config.py) if it's intentional."
        )


_check_for_prefix_typos()

# Instantiated once at import time and imported by name (`from
# app.core.config import settings`) rather than re-constructed per use —
# this *is* the app's configuration, not a factory for building one.
settings = Settings()
