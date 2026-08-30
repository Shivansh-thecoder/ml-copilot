"""The MCP server: exposes the ML engine's capabilities as MCP tools.

Every tool here is a plain, undecorated function imported from `tools/`
-- this file's only job is registering them with FastMCP; none of the
actual dataset-loading, EDA, training, or persistence logic lives here
(see `tools/` and `ml_engine/`).

A client discovers tools dynamically via `list_tools()` rather than a
hardcoded list -- registering a new tool here requires zero client-side
changes; it's picked up the next time a client calls `list_tools()`.

RUNNING THIS
----------------
    backend/mcp/.venv/Scripts/python.exe backend/mcp/server.py

Must be run with `backend/mcp/.venv`'s interpreter, which has
pandas/numpy/scikit-learn/scipy/joblib installed alongside `mcp` (see
backend/mcp/requirements.txt) -- not `backend/.venv`.

Run from anywhere; Python puts the SCRIPT's own directory (`backend/mcp`)
on `sys.path[0]`, which is what lets `import tools...`/`import
ml_engine...` below resolve correctly. (Running instead as `python -c
"..."` or from a cwd of `backend/` -- rather than executing this file as
a script -- can shadow the installed `mcp` SDK with this package, since
both are named `mcp`; see README.md's "naming-collision gotcha" if you
hit that.)
"""

from mcp.server.fastmcp import FastMCP

from tools.dataset_tools import inspect_dataset, profile_dataset
from tools.model_management_tools import get_experiment_results, list_saved_models, save_model
from tools.modeling_tools import train_models, tune_model
from tools.prediction_tools import get_model_schema, predict_with_model
from tools.target_tools import detect_target

mcp = FastMCP("ml-copilot")

# Each call introspects the function's name, type hints, and docstring to
# build that tool's name, input schema, and description -- no JSON schema
# is hand-written anywhere in this file or in `tools/`.
mcp.tool()(inspect_dataset)
mcp.tool()(profile_dataset)
mcp.tool()(detect_target)
mcp.tool()(train_models)
mcp.tool()(tune_model)
mcp.tool()(save_model)
mcp.tool()(get_experiment_results)
mcp.tool()(list_saved_models)
mcp.tool()(get_model_schema)
mcp.tool()(predict_with_model)

if __name__ == "__main__":
    # stdio: this is a local, single-machine server started as a
    # subprocess by its client (backend/langgraph/mcp_client/). No host/
    # port to configure, nothing reachable from outside this machine --
    # the simplest transport for what this server needs.
    mcp.run(transport="stdio")
