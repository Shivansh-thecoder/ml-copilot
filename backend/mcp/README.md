# MCP server

A local MCP server exposing the ML engine (`ml_engine/`) as tools:
`inspect_dataset`, `profile_dataset`, `detect_target`, `train_models`,
`tune_model`, `save_model`, `get_experiment_results`, `list_saved_models`,
`get_model_schema`, `predict_with_model`.

## Why this runs in its own environment

`backend/mcp/.venv` is separate from `backend/.venv` (the FastAPI app)
because their dependencies genuinely conflict: `mcp`'s SSE transport
(via `sse-starlette`) needs `starlette>=0.49`, while `fastapi==0.115.6`
pins `starlette<0.42.0`. No single `starlette` version satisfies both.

The MCP **client** (`backend/langgraph/mcp_client/`) doesn't have this
problem — `mcp.client.*` never imports `starlette` at all — so it lives
in `backend/.venv` alongside FastAPI/LangGraph, installed with
`sse-starlette==1.6.1` pinned specifically (an old version whose own
`starlette` requirement is unconstrained, so it doesn't drag in the
conflicting one). Only the **server** needs its own environment.

## Setup

```bash
cd backend
python -m venv mcp/.venv
mcp/.venv/Scripts/pip install -r mcp/requirements.txt
```

Nothing needs to be started manually — `backend/langgraph/mcp_client/`
spawns `server.py` as a subprocess, once per tool call, using
`mcp/.venv`'s interpreter.

To run it directly (e.g. for manual testing):

```bash
backend/mcp/.venv/Scripts/python.exe backend/mcp/server.py
```

## A naming-collision gotcha

This directory is named `mcp` — the same name as the installed `mcp` SDK
package. If `backend/` (this directory's *parent*) ever ends up on
`sys.path`, `import mcp` resolves to `backend/mcp/__init__.py` instead of
the real SDK, breaking every `from mcp...` import in that process.

This bites in two situations, both confirmed directly while building
this server:

- Running a one-off script with `python -c "..."` from a cwd of
  `backend/` — `-c` puts cwd on `sys.path[0]`.
- Any process that also needs to import `backend/app/` (which requires
  `backend/` on `sys.path` to resolve `app.*` imports at all) — see
  `backend/app/api/routers/copilot.py`'s `_import_langgraph_pieces()`
  for how that specific case is handled (temporarily removing `backend/`
  from `sys.path` for the one import that needs to avoid it, then
  restoring it).

Running `server.py` as a script (as documented above) is safe on its
own — Python puts the *script's own directory* (`backend/mcp`, not
`backend`) on `sys.path[0]`.

## Tool contract

Every tool takes a real `dataset_id` (the same id `GET /datasets`
returns) and resolves it to an actual uploaded CSV via
`ml_engine/dataset_io.py`, which reads `backend/app.db`'s `datasets`
table directly with `sqlite3` — this package never imports
`app.core.config`, keeping it independent of the FastAPI app's settings
and its dependencies.

`train_models` and `tune_model` persist every candidate's fitted
pipeline (preprocessing + model, as one joblib artifact) to
`storage/models/`, and log one row per candidate to `storage/experiments.db`
— a second, purpose-built SQLite file, not `app.db` (which Alembic
owns). `save_model` promotes one candidate's artifact to
`storage/saved/{dataset_id}.joblib` as the dataset's current saved model.

`predict_with_model` reloads that saved artifact and calls `.predict()`
on it directly. It never rebuilds preprocessing: the artifact's
preprocessing step carries the medians, scaling and one-hot category
ordering learned during training, and refitting those on the incoming
rows would silently change what each feature position means. The same
artifact is also what `get_model_schema` reads the expected feature
columns off, so no layer above the ML engine knows any dataset's column
names.

## Transport

`stdio` — the server communicates over its own process's stdin/stdout.
Appropriate for a local, single-machine server started as a subprocess
by its one client; no host/port to configure, nothing reachable from
outside the machine.

## Testing

```bash
cd backend/mcp
.venv/Scripts/python.exe -m pytest tests/ -v
```

Covers the ML engine directly (dataset inspection, target detection,
leakage prevention, training, tuning, diagnostics, persistence) and the
MCP tool wrappers, plus one test that spawns the real server as a
subprocess and verifies tool discovery over the actual protocol.
