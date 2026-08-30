"""HTTP-level tests for /models.

The MCP call is stubbed: what's under test here is the router's own job
-- passing arguments through, and turning an MCP tool's structured
`{"error": ...}` result into the right HTTP status. The ML behaviour
behind that boundary is covered by backend/mcp/tests/test_prediction.py.
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.api.routers import models as models_router
from app.main import app

client = TestClient(app)


@pytest.fixture
def mcp_calls(monkeypatch):
    """Record every MCP call and return a queued canned result."""
    calls = []
    responses = {}

    def fake_call(name: str, arguments: dict) -> str:
        calls.append((name, arguments))
        return json.dumps(responses[name])

    monkeypatch.setattr(models_router, "call_mcp_tool_sync", fake_call)
    return {"calls": calls, "responses": responses}


def test_list_models(mcp_calls):
    mcp_calls["responses"]["list_saved_models"] = {
        "count": 1,
        "models": [{"dataset_id": "ds-1", "model_name": "random_forest"}],
    }

    response = client.get("/models")

    assert response.status_code == 200
    assert response.json()["models"][0]["dataset_id"] == "ds-1"
    assert mcp_calls["calls"] == [("list_saved_models", {})]


def test_get_schema_passes_the_dataset_id_through(mcp_calls):
    mcp_calls["responses"]["get_model_schema"] = {
        "target_column": "churned",
        "features": [{"name": "age", "kind": "numeric"}],
    }

    response = client.get("/models/ds-1/schema")

    assert response.status_code == 200
    assert response.json()["features"][0]["name"] == "age"
    assert mcp_calls["calls"] == [("get_model_schema", {"dataset_id": "ds-1"})]


def test_schema_for_a_dataset_with_no_saved_model_is_404(mcp_calls):
    mcp_calls["responses"]["get_model_schema"] = {
        "error": "no_saved_model",
        "message": "No saved model for dataset 'ds-9'.",
    }

    response = client.get("/models/ds-9/schema")

    assert response.status_code == 404
    assert "No saved model" in response.json()["detail"]


def test_predict_returns_predictions(mcp_calls):
    mcp_calls["responses"]["predict_with_model"] = {
        "task_type": "classification",
        "target_column": "churned",
        "predictions": [
            {"row": 0, "prediction": 1, "probabilities": {"0": 0.2, "1": 0.8}, "confidence": 0.8}
        ],
        "ignored_columns": [],
        "warnings": [],
    }

    response = client.post("/models/ds-1/predict", json={"rows": [{"age": 30, "plan": "pro"}]})

    assert response.status_code == 200
    assert response.json()["predictions"][0]["confidence"] == 0.8
    assert mcp_calls["calls"] == [
        ("predict_with_model", {"dataset_id": "ds-1", "rows": [{"age": 30, "plan": "pro"}]})
    ]


def test_predict_with_a_missing_feature_is_422(mcp_calls):
    mcp_calls["responses"]["predict_with_model"] = {
        "error": "missing_features",
        "message": "Row 0 is missing required feature(s): plan.",
        "row": 0,
        "missing": ["plan"],
    }

    response = client.post("/models/ds-1/predict", json={"rows": [{"age": 30}]})

    assert response.status_code == 422
    assert "plan" in response.json()["detail"]


def test_predict_with_an_invalid_value_is_422(mcp_calls):
    mcp_calls["responses"]["predict_with_model"] = {
        "error": "invalid_feature_value",
        "message": "Row 0: value 'old' for feature 'age' is not a number.",
        "row": 0,
        "column": "age",
    }

    response = client.post("/models/ds-1/predict", json={"rows": [{"age": "old", "plan": "pro"}]})

    assert response.status_code == 422
    assert "not a number" in response.json()["detail"]


def test_predict_with_no_rows_never_reaches_mcp(mcp_calls):
    response = client.post("/models/ds-1/predict", json={"rows": []})

    assert response.status_code == 422
    assert mcp_calls["calls"] == []


def test_mcp_failure_surfaces_as_bad_gateway(monkeypatch):
    def boom(name: str, arguments: dict) -> str:
        raise RuntimeError("MCP server did not start")

    monkeypatch.setattr(models_router, "call_mcp_tool_sync", boom)

    response = client.get("/models")

    assert response.status_code == 502
    assert "MCP server did not start" in response.json()["detail"]
