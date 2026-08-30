from tests.conftest import no_tool_call_message, tool_call_message

from nodes.copilot_tools import (
    detect_target_stage,
    inspect_stage,
    profile_stage,
    retrieve_results_stage,
    save_stage,
    train_stage,
    tune_stage,
)


def test_inspect_stage_success(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_llm([tool_call_message("inspect_dataset", {"dataset_id": "ds-test"})])
    mock_mcp_call.script("inspect_dataset", {"n_rows": 100, "n_columns": 5})

    update = inspect_stage(base_state)

    assert "errors" not in update
    assert update["completed_steps"] == ["inspect"]
    assert update["tool_results"][0]["result"]["n_rows"] == 100
    assert update["tool_call_count"] == 1
    assert update["iteration_count"] == 1


def test_profile_stage_success(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_llm([tool_call_message("profile_dataset", {"dataset_id": "ds-test"})])
    mock_mcp_call.script("profile_dataset", {"outlier_signals": {}, "top_correlations": []})

    update = profile_stage(base_state)

    assert "errors" not in update
    assert update["completed_steps"] == ["profile"]


def test_detect_target_stage_resolved(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_llm([tool_call_message("detect_target", {"dataset_id": "ds-test"})])
    mock_mcp_call.script(
        "detect_target",
        {"status": "resolved", "target_column": "churned", "task_type": "classification", "candidates": []},
    )

    update = detect_target_stage(base_state)

    assert update["target_column"] == "churned"
    assert update["task_type"] == "classification"
    assert update["needs_clarification"] is False


def test_detect_target_stage_needs_clarification(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_llm([tool_call_message("detect_target", {"dataset_id": "ds-test"})])
    mock_mcp_call.script(
        "detect_target",
        {
            "status": "needs_clarification",
            "target_column": None,
            "task_type": None,
            "candidates": [{"column": "churned", "score": 0.4}, {"column": "cancellation_date", "score": 0.3}],
            "reasoning": "no confident match",
        },
    )

    update = detect_target_stage(base_state)

    assert update["needs_clarification"] is True
    assert "churned" in update["clarification_question"]
    assert "cancellation_date" in update["clarification_question"]


def test_train_stage_success(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    base_state["target_column"] = "churned"
    base_state["task_type"] = "classification"
    mock_llm([tool_call_message("train_models", {"dataset_id": "ds-test", "target_column": "churned"})])
    mock_mcp_call.script(
        "train_models",
        {"run_id": "run-1", "best_model": "logistic_regression", "results": [], "primary_metric": "f1"},
    )

    update = train_stage(base_state)

    assert update["run_id"] == "run-1"
    assert update["selected_model"] == "logistic_regression"


def test_tune_stage_failure_is_recoverable(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    base_state["selected_model"] = "logistic_regression"
    base_state["run_id"] = "run-1"
    # both retry attempts fail
    mock_llm([no_tool_call_message(), no_tool_call_message()])

    update = tune_stage(base_state)

    assert "errors" in update
    assert update["errors"][0]["recoverable"] is True


def test_save_stage_failure_is_recoverable(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    base_state["selected_model"] = "logistic_regression"
    base_state["run_id"] = "run-1"
    mock_llm([no_tool_call_message(), no_tool_call_message()])

    update = save_stage(base_state)

    assert "errors" in update
    assert update["errors"][0]["recoverable"] is True


def test_retrieve_results_stage_success(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_llm([tool_call_message("get_experiment_results", {"dataset_id": "ds-test"})])
    mock_mcp_call.script("get_experiment_results", {"dataset_id": "ds-test", "run_count": 1, "runs": [], "saved_model": None})

    update = retrieve_results_stage(base_state)

    assert "errors" not in update


def test_stage_retries_on_tool_error_then_succeeds(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    """First tool call errors (e.g. bad args), second attempt succeeds --
    exercises run_tool_stage's retry loop directly."""
    mock_llm(
        [
            tool_call_message("inspect_dataset", {"dataset_id": "wrong-id"}),
            tool_call_message("inspect_dataset", {"dataset_id": "ds-test"}),
        ]
    )
    mock_mcp_call.script("inspect_dataset", [Exception("no such dataset"), {"n_rows": 50}])

    update = inspect_stage(base_state)

    assert "errors" not in update
    assert len(mock_mcp_call.calls) == 2  # both attempts actually called the tool


def test_stage_fails_after_exhausting_retries(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_llm([no_tool_call_message(), no_tool_call_message()])

    update = inspect_stage(base_state)

    assert "errors" in update
    assert update["consecutive_failures"] == 1


def test_tool_reported_error_triggers_retry_path(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_llm([tool_call_message("inspect_dataset", {"dataset_id": "does-not-exist"}), tool_call_message("inspect_dataset", {"dataset_id": "ds-test"})])
    # simulate: first "call" returns a tool-reported error dict, second succeeds.
    # mock_mcp_call.script only supports one static script per name, so
    # emulate via a stateful closure instead.
    import nodes.copilot_tools as copilot_tools

    attempts = {"n": 0}

    def fake_call(name, args):
        attempts["n"] += 1
        import json

        if attempts["n"] == 1:
            return json.dumps({"error": "dataset_not_found", "message": "no such dataset"})
        return json.dumps({"n_rows": 10})

    from unittest.mock import patch

    with patch.object(copilot_tools, "call_mcp_tool_sync", fake_call):
        update = inspect_stage(base_state)

    assert "errors" not in update
    assert attempts["n"] == 2
