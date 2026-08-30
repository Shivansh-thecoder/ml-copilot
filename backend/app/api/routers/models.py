"""HTTP surface for saved models: listing them, describing what features
they expect, and predicting on new rows.

Every endpoint here goes Frontend -> FastAPI -> MCP client -> MCP server
-> ML engine. This process never imports pandas or scikit-learn (they
live only in `backend/mcp/.venv`) and never touches a .joblib file --
it calls MCP tools by name and passes their structured results through.
"""

import json

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.langgraph_bridge import langgraph_imports

# See app/api/langgraph_bridge.py: `backend/mcp/` shadows the installed
# `mcp` SDK that this client imports, while `backend/` is on sys.path.
with langgraph_imports():
    from mcp_client.client import call_mcp_tool_sync

router = APIRouter(prefix="/models", tags=["models"])

# MCP tool errors are structured results, not exceptions -- map each
# error code onto the HTTP status that actually describes it, so the
# frontend can rely on normal HTTP semantics.
_ERROR_STATUS = {
    "no_saved_model": 404,
    "artifact_missing": 404,
    "unreadable_model": 422,
    "missing_features": 422,
    "invalid_feature_value": 422,
    "invalid_request": 422,
}


def _call_tool(name: str, arguments: dict) -> dict:
    try:
        raw = call_mcp_tool_sync(name, arguments)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"MCP tool '{name}' failed: {exc}") from exc

    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=502, detail=f"MCP tool '{name}' returned a non-JSON result."
        ) from exc

    if isinstance(result, dict) and "error" in result:
        status = _ERROR_STATUS.get(result["error"], 400)
        raise HTTPException(status_code=status, detail=result.get("message", result["error"]))
    return result


class PredictRequest(BaseModel):
    rows: list[dict]


@router.get("")
def list_saved_models() -> dict:
    """Every dataset that currently has a saved, ready-to-predict model."""
    return _call_tool("list_saved_models", {})


@router.get("/{dataset_id}/schema")
def get_model_schema(dataset_id: str) -> dict:
    """The saved model's details plus the feature columns it expects.

    The frontend builds its prediction form from this -- the columns come
    from the fitted pipeline inside the saved artifact, so no dataset's
    column names are known to (or hardcoded in) any layer above the ML
    engine.
    """
    return _call_tool("get_model_schema", {"dataset_id": dataset_id})


@router.post("/{dataset_id}/predict")
def predict(dataset_id: str, data: PredictRequest) -> dict:
    """Predict on new feature rows using the dataset's saved model."""
    if not data.rows:
        raise HTTPException(status_code=422, detail="Provide at least one row to predict on.")
    return _call_tool("predict_with_model", {"dataset_id": dataset_id, "rows": data.rows})
