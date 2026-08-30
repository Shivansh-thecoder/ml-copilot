"""Integration tests: the thin MCP tool wrappers (dataset_id in, structured
dict out) and the real FastMCP server's tool discovery.

These call the plain Python functions in `tools/` directly, not over the
actual MCP stdio protocol -- these tests are about the tool functions'
own logic and error handling, which is what unit tests should cover: no
live OpenAI calls, no subprocess (except the discovery test below, which
needs a real subprocess for a different reason).
"""

from tools.dataset_tools import inspect_dataset, profile_dataset
from tools.model_management_tools import get_experiment_results, list_saved_models, save_model
from tools.modeling_tools import train_models, tune_model
from tools.prediction_tools import get_model_schema, predict_with_model
from tools.target_tools import detect_target


def test_inspect_dataset_tool_success(seeded_dataset):
    result = inspect_dataset(seeded_dataset["dataset_id"])
    assert result["n_rows"] == len(seeded_dataset["df"])
    assert "error" not in result


def test_inspect_dataset_tool_nonexistent_dataset():
    result = inspect_dataset("does-not-exist")
    assert result["error"] == "dataset_not_found"


def test_profile_dataset_tool_uses_stored_target_column(seeded_dataset):
    result = profile_dataset(seeded_dataset["dataset_id"])
    assert result["target_distribution"] is not None  # "churned" resolved from the dataset row


def test_detect_target_tool_uses_stored_target_column(seeded_dataset):
    result = detect_target(seeded_dataset["dataset_id"])
    assert result["status"] == "resolved"
    assert result["target_column"] == "churned"


def test_train_models_tool_end_to_end(seeded_dataset):
    result = train_models(seeded_dataset["dataset_id"], cv_folds=2)
    assert "error" not in result
    assert result["best_model"] is not None
    assert len(result["results"]) == 3


def test_train_models_tool_nonexistent_dataset():
    result = train_models("does-not-exist")
    assert result["error"] == "dataset_not_found"


def test_full_train_save_retrieve_flow(seeded_dataset):
    dataset_id = seeded_dataset["dataset_id"]

    train_result = train_models(dataset_id, cv_folds=2)
    assert "error" not in train_result
    run_id = train_result["run_id"]
    best_model = train_result["best_model"]

    save_result = save_model(dataset_id, run_id=run_id, model_name=best_model)
    assert "error" not in save_result
    assert save_result["metadata"]["model_name"] == best_model

    experiment_results = get_experiment_results(dataset_id)
    assert experiment_results["run_count"] == 3  # one row per candidate model
    assert experiment_results["saved_model"]["model_name"] == best_model


def test_save_model_unknown_run_id_returns_error(seeded_dataset):
    result = save_model(seeded_dataset["dataset_id"], run_id="no-such-run", model_name="random_forest")
    assert result["error"] == "save_failed"


def test_tune_model_tool_end_to_end(seeded_dataset):
    result = tune_model(seeded_dataset["dataset_id"], model_name="logistic_regression", n_iter=3, cv_folds=2)
    assert "error" not in result
    assert "best_params" in result


def test_tune_model_unknown_model_returns_error(seeded_dataset):
    result = tune_model(seeded_dataset["dataset_id"], model_name="not_a_real_model", n_iter=3, cv_folds=2)
    assert result["error"] == "unknown_model"


def test_predict_round_trip_through_the_mcp_tools(seeded_dataset):
    """train -> save -> read the schema -> predict, entirely through the
    tool layer, the same sequence an MCP client drives."""
    dataset_id = seeded_dataset["dataset_id"]

    train_result = train_models(dataset_id, cv_folds=2)
    save_model(dataset_id, run_id=train_result["run_id"], model_name=train_result["best_model"])

    schema = get_model_schema(dataset_id)
    assert "error" not in schema
    # The seeded dataset's features, discovered from the saved pipeline --
    # never named by the tool layer or anything above it.
    assert [f["name"] for f in schema["features"]] == ["age", "income", "plan"]
    assert schema["target_column"] == "churned"

    row = {feature["name"]: ("basic" if feature["name"] == "plan" else 1) for feature in schema["features"]}
    result = predict_with_model(dataset_id, [row])
    assert "error" not in result
    assert result["predictions"][0]["prediction"] in (0, 1)
    assert result["predictions"][0]["confidence"] > 0


def test_predict_tool_reports_a_missing_feature(seeded_dataset):
    dataset_id = seeded_dataset["dataset_id"]
    train_result = train_models(dataset_id, cv_folds=2)
    save_model(dataset_id, run_id=train_result["run_id"], model_name=train_result["best_model"])

    result = predict_with_model(dataset_id, [{"age": 30, "income": 50000}])
    assert result["error"] == "missing_features"
    assert result["missing"] == ["plan"]


def test_predict_tool_reports_an_invalid_value(seeded_dataset):
    dataset_id = seeded_dataset["dataset_id"]
    train_result = train_models(dataset_id, cv_folds=2)
    save_model(dataset_id, run_id=train_result["run_id"], model_name=train_result["best_model"])

    result = predict_with_model(dataset_id, [{"age": "old", "income": 50000, "plan": "pro"}])
    assert result["error"] == "invalid_feature_value"
    assert result["column"] == "age"


def test_prediction_tools_without_a_saved_model_return_errors():
    assert get_model_schema("no-such-dataset")["error"] == "no_saved_model"
    assert predict_with_model("no-such-dataset", [{"a": 1}])["error"] == "no_saved_model"


def test_list_saved_models_tool(seeded_dataset):
    assert list_saved_models() == {"count": 0, "models": []}

    dataset_id = seeded_dataset["dataset_id"]
    train_result = train_models(dataset_id, cv_folds=2)
    save_model(dataset_id, run_id=train_result["run_id"], model_name=train_result["best_model"])

    listed = list_saved_models()
    assert listed["count"] == 1
    assert listed["models"][0]["dataset_id"] == dataset_id


def test_get_experiment_results_no_history_returns_empty():
    result = get_experiment_results("dataset-with-no-runs")
    assert result["run_count"] == 0
    assert result["saved_model"] is None


def test_mcp_server_discovers_all_tools_dynamically():
    """The real FastMCP instance from server.py -- confirms every tool is
    actually registered and discoverable, matching what
    backend/langgraph/mcp_client/ would see.

    Runs server.py's discovery in a FRESH subprocess rather than importing
    it into this test process directly: this test file's own directory
    (backend/mcp) shares its name with the installed `mcp` SDK package
    (see README.md's naming-collision notes), and pytest's own import
    machinery ends up binding sys.modules["mcp"] to backend/mcp/__init__.py
    before this test runs -- confirmed by inspection, not just theory.
    A subprocess gets a clean interpreter with no such pre-existing
    sys.modules entry, exactly like running `server.py` directly (verified
    manually) does.
    """
    import json
    import subprocess
    import sys
    from pathlib import Path

    mcp_dir = Path(__file__).resolve().parent.parent
    script = (
        "import asyncio, sys; sys.path.insert(0, r'{}'); import server\n"
        "tools = asyncio.run(server.mcp.list_tools())\n"
        "import json; print(json.dumps(sorted(t.name for t in tools)))"
    ).format(mcp_dir)

    result = subprocess.run(
        [sys.executable, "-c", script],
        # NOT mcp_dir.parent ("backend/") -- that IS the dangerous cwd (see
        # README.md's naming-collision notes: `-c` puts cwd on sys.path[0],
        # and "backend" contains a directory literally named "mcp"). The
        # repo root has no such subdirectory, so it's safe.
        cwd=str(mcp_dir.parent.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr

    tool_names = set(json.loads(result.stdout.strip().splitlines()[-1]))
    assert tool_names == {
        "inspect_dataset",
        "profile_dataset",
        "detect_target",
        "train_models",
        "tune_model",
        "save_model",
        "get_experiment_results",
        "list_saved_models",
        "get_model_schema",
        "predict_with_model",
    }
