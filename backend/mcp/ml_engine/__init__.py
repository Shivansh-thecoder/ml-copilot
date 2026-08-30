"""The ML engine -- deterministic pandas/scikit-learn computation, exposed through MCP tools.

Every function in this package takes and returns plain data (DataFrames,
dicts, primitives) -- never an LLM, a chat model, or MCP itself. That
split matters: this code is what actually DOES the ML work (loading
data, computing statistics, fitting models, scoring them); `backend/mcp/
tools/` is the thin layer that exposes these functions as MCP tools
keyed by `dataset_id`; the LLM (via LangGraph, via the MCP client) only
ever chooses WHICH function to call and interprets its structured
result -- it never computes a metric or fits a model itself.

Runs inside `backend/mcp/.venv` (see backend/mcp/requirements.txt) --
the same isolated environment `backend/mcp/server.py` already runs in,
now extended with pandas/numpy/scikit-learn/scipy/joblib. Neither
`backend/.venv` (FastAPI + LangGraph) nor `backend/app/` needs these.

WHY THIS PACKAGE NEVER IMPORTS `backend/app/`
---------------------------------------------------
`backend/langgraph/` stays import-independent of `backend/app/`, and
this package follows the same rule: `dataset_io.py` reads `app.db`'s
`datasets` table directly with the standard library's `sqlite3` module
instead of importing `app.core.config`/SQLAlchemy/SQLModel. This keeps
the ML engine usable without pulling in the FastAPI app's dependencies,
and avoids coupling two independently-runnable pieces of the system.
"""
