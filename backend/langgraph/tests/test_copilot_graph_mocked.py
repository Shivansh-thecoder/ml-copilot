"""Full-graph tests -- LLM AND the MCP client both mocked (no subprocess,
no network). Fast, deterministic, run on every test invocation. See
test_copilot_graph_integration.py for the smaller set of tests that use
the REAL MCP client/server/ML engine instead.
"""

from langchain_core.messages import AIMessage
from langgraph.types import Command

from graphs.copilot_graph import build_copilot_graph
from nodes.copilot_common import MAX_ITERATIONS
from schemas.copilot import CopilotPlan, FinalNarrative
from tests.conftest import tool_call_message

DATASET_ID = "ds-graph-test"


def _plan():
    return CopilotPlan(
        objective="Build a model to predict customer churn.",
        dataset_id=DATASET_ID,
        target_column=None,
        task_type=None,
        intended_analysis_steps=["Inspect", "Profile"],
        intended_modeling_steps=["Train", "Evaluate"],
    )


def _initial_state(objective="Build a model to predict customer churn."):
    return {
        "objective": objective,
        "dataset_id": DATASET_ID,
        "plan": None,
        "target_column": None,
        "task_type": None,
        "messages": [],
        "tool_results": [],
        "current_step": "start",
        "completed_steps": [],
        "errors": [],
        "needs_clarification": False,
        "clarification_question": None,
        "run_id": None,
        "selected_model": None,
        "tuning_performed": False,
        "tuning_reason": "",
        "saved_model_info": None,
        "final_response": None,
        "iteration_count": 0,
        "tool_call_count": 0,
        "consecutive_failures": 0,
    }


TRAIN_RESULT = {
    "run_id": "run-1",
    "best_model": "logistic_regression",
    "primary_metric": "f1",
    "primary_metric_reason": "balances precision/recall",
    "task_type": "classification",
    "results": [
        {
            "model": "logistic_regression",
            "cv_mean": {"f1": 0.95},
            "cv_std": {"f1": 0.01},
            "test_metrics": {"f1": 0.94, "accuracy": 0.95},
            "diagnostic": {"diagnostic": "no_strong_signal", "evidence": "train=0.96, val=0.95"},
        }
    ],
}


def test_normal_classification_flow_no_tuning(base_state, mock_llm, mock_mcp_tools, mock_mcp_call, monkeypatch):
    """planner -> inspect -> profile -> detect_target(resolved) -> train ->
    tune_decision(skip, strong baseline) -> save -> retrieve_results -> finalize."""
    mock_mcp_call.script("inspect_dataset", {"n_rows": 200, "n_columns": 5})
    mock_mcp_call.script("profile_dataset", {"outlier_signals": {}, "top_correlations": []})
    mock_mcp_call.script(
        "detect_target",
        {"status": "resolved", "target_column": "churned", "task_type": "classification", "candidates": []},
    )
    mock_mcp_call.script("train_models", TRAIN_RESULT)
    mock_mcp_call.script("save_model", {"saved_model_path": "/storage/saved/ds-graph-test.joblib", "metadata": {}})
    mock_mcp_call.script(
        "get_experiment_results", {"dataset_id": DATASET_ID, "run_count": 1, "runs": [], "saved_model": None}
    )

    mock_llm(
        [
            _plan(),
            tool_call_message("inspect_dataset", {"dataset_id": DATASET_ID}),
            tool_call_message("profile_dataset", {"dataset_id": DATASET_ID}),
            tool_call_message("detect_target", {"dataset_id": DATASET_ID}),
            tool_call_message("train_models", {"dataset_id": DATASET_ID, "target_column": "churned", "task_type": "classification"}),
            # (no tune -- strong baseline skips it)
            tool_call_message("save_model", {"dataset_id": DATASET_ID, "run_id": "run-1", "model_name": "logistic_regression"}),
            tool_call_message("get_experiment_results", {"dataset_id": DATASET_ID}),
            FinalNarrative(comparison_summary="Logistic regression was the only strong candidate.", caveats=["Single dataset run."]),
        ]
    )

    graph = build_copilot_graph()
    config = {"configurable": {"thread_id": "test-normal-classification"}}
    result = graph.invoke(_initial_state(), config)

    assert "__interrupt__" not in result
    response = result["final_response"]
    assert response.success is True
    assert response.selected_model == "logistic_regression"
    assert response.tuning_performed is False
    assert response.saved_model_location == "/storage/saved/ds-graph-test.joblib"
    assert result["errors"] == []


def test_clarification_pause_and_resume(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_mcp_call.script("inspect_dataset", {"n_rows": 200, "n_columns": 5})
    mock_mcp_call.script("profile_dataset", {"outlier_signals": {}, "top_correlations": []})
    mock_mcp_call.script(
        "detect_target",
        [
            {
                "status": "needs_clarification",
                "target_column": None,
                "task_type": None,
                "candidates": [{"column": "churned", "score": 0.4}, {"column": "cancelled", "score": 0.35}],
            },
            {"status": "resolved", "target_column": "churned", "task_type": "classification", "candidates": []},
        ],
    )
    mock_mcp_call.script("train_models", TRAIN_RESULT)
    mock_mcp_call.script("save_model", {"saved_model_path": "/x.joblib", "metadata": {}})
    mock_mcp_call.script("get_experiment_results", {"dataset_id": DATASET_ID, "run_count": 1, "runs": [], "saved_model": None})

    mock_llm(
        [
            _plan(),
            tool_call_message("inspect_dataset", {"dataset_id": DATASET_ID}),
            tool_call_message("profile_dataset", {"dataset_id": DATASET_ID}),
            tool_call_message("detect_target", {"dataset_id": DATASET_ID}),  # -> needs_clarification, graph pauses
            tool_call_message("detect_target", {"dataset_id": DATASET_ID, "target_column": "churned"}),  # after resume
            tool_call_message("train_models", {"dataset_id": DATASET_ID, "target_column": "churned", "task_type": "classification"}),
            tool_call_message("save_model", {"dataset_id": DATASET_ID, "run_id": "run-1", "model_name": "logistic_regression"}),
            tool_call_message("get_experiment_results", {"dataset_id": DATASET_ID}),
            FinalNarrative(comparison_summary="Resolved after clarification.", caveats=[]),
        ]
    )

    graph = build_copilot_graph()
    config = {"configurable": {"thread_id": "test-clarification"}}

    paused = graph.invoke(_initial_state(), config)
    assert "__interrupt__" in paused
    assert "churned" in paused["__interrupt__"][0].value["question"]

    resumed = graph.invoke(Command(resume="churned"), config)
    assert "__interrupt__" not in resumed
    assert resumed["target_column"] == "churned"
    assert resumed["final_response"].success is True


def test_missing_dataset_aborts_to_finalize_with_failure(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    mock_mcp_call.script("inspect_dataset", [Exception("mcp tool error"), Exception("mcp tool error")])

    mock_llm(
        [
            _plan(),
            tool_call_message("inspect_dataset", {"dataset_id": "does-not-exist"}),
            tool_call_message("inspect_dataset", {"dataset_id": "does-not-exist"}),  # retry, also fails
        ]
    )

    graph = build_copilot_graph()
    config = {"configurable": {"thread_id": "test-missing-dataset"}}
    result = graph.invoke(_initial_state(), config)

    response = result["final_response"]
    assert response.success is False
    assert response.failure_reason is not None
    assert result["completed_steps"] == ["planner"]  # never reached "inspect"


def test_max_iteration_limit_terminates_workflow(base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    """Force iteration_count already at the limit before the graph even
    starts its first tool-calling stage -- confirms check_limits() (called
    at the top of every stage) actually stops the workflow rather than
    proceeding, and the graph still reaches a clean finalize/END."""
    mock_llm([_plan()])  # planner runs; inspect_stage should refuse before calling the model again

    graph = build_copilot_graph()
    config = {"configurable": {"thread_id": "test-max-iterations"}}
    state = _initial_state()
    state["iteration_count"] = MAX_ITERATIONS

    result = graph.invoke(state, config)

    response = result["final_response"]
    assert response.success is False
    assert "iterations" in response.failure_reason.lower()
