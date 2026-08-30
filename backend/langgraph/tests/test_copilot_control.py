from nodes.copilot_common import MAX_CONSECUTIVE_FAILURES, MAX_ITERATIONS, MAX_TOOL_CALLS
from nodes.copilot_control import check_limits, finalize_node, tune_decision_node


def _train_result(diagnostic: str, f1: float = 0.75, r2: float | None = None, task_type: str = "classification"):
    metric = "r2" if task_type == "regression" else "f1"
    cv_mean = {"f1": f1} if task_type == "classification" else {"rmse": 2.0, "r2": r2 if r2 is not None else 0.5}
    return {
        "primary_metric": metric if task_type == "classification" else "rmse",
        "task_type": task_type,
        "results": [
            {
                "model": "logistic_regression",
                "cv_mean": cv_mean,
                "test_metrics": {"f1": f1} if task_type == "classification" else {"rmse": 2.0, "r2": r2},
                "diagnostic": {"diagnostic": diagnostic, "evidence": "train=0.9, val=0.7"},
            }
        ],
    }


def test_check_limits_none_when_under_bounds(base_state):
    assert check_limits(base_state) is None


def test_check_limits_max_iterations(base_state):
    base_state["iteration_count"] = MAX_ITERATIONS
    error = check_limits(base_state)
    assert error is not None
    assert "iterations" in error["message"].lower()


def test_check_limits_max_tool_calls(base_state):
    base_state["tool_call_count"] = MAX_TOOL_CALLS
    error = check_limits(base_state)
    assert error is not None
    assert "tool calls" in error["message"].lower()


def test_check_limits_max_consecutive_failures(base_state):
    base_state["consecutive_failures"] = MAX_CONSECUTIVE_FAILURES
    error = check_limits(base_state)
    assert error is not None
    assert "failures" in error["message"].lower()


def test_tune_decision_overfitting_tunes(base_state):
    base_state["selected_model"] = "logistic_regression"
    base_state["tool_results"] = [{"stage": "train", "result": _train_result("possible_overfitting")}]

    update = tune_decision_node(base_state)

    assert update["tuning_performed"] is True
    assert "overfit" in update["tuning_reason"].lower()


def test_tune_decision_underfitting_skips(base_state):
    base_state["selected_model"] = "logistic_regression"
    base_state["tool_results"] = [{"stage": "train", "result": _train_result("possible_underfitting")}]

    update = tune_decision_node(base_state)

    assert update["tuning_performed"] is False
    assert "underfit" in update["tuning_reason"].lower()


def test_tune_decision_strong_baseline_skips(base_state):
    base_state["selected_model"] = "logistic_regression"
    base_state["tool_results"] = [{"stage": "train", "result": _train_result("no_strong_signal", f1=0.95)}]

    update = tune_decision_node(base_state)

    assert update["tuning_performed"] is False
    assert "strongly" in update["tuning_reason"].lower()


def test_tune_decision_weak_baseline_tunes(base_state):
    base_state["selected_model"] = "logistic_regression"
    base_state["tool_results"] = [{"stage": "train", "result": _train_result("no_strong_signal", f1=0.65)}]

    update = tune_decision_node(base_state)

    assert update["tuning_performed"] is True
    assert "room to improve" in update["tuning_reason"].lower()


def test_tune_decision_regression_uses_r2_for_strength(base_state):
    base_state["selected_model"] = "logistic_regression"
    base_state["tool_results"] = [
        {"stage": "train", "result": _train_result("no_strong_signal", r2=0.95, task_type="regression")}
    ]

    update = tune_decision_node(base_state)

    assert update["tuning_performed"] is False  # r2=0.95 is "strong"


def test_finalize_node_failure_when_no_train_result(base_state, mock_llm):
    base_state["errors"] = [{"stage": "inspect", "message": "dataset not found", "recoverable": False}]

    update = finalize_node(base_state)

    response = update["final_response"]
    assert response.success is False
    assert "dataset not found" in response.failure_reason


def test_finalize_node_success_grounds_numbers_from_state(base_state, mock_llm):
    from schemas.copilot import FinalNarrative

    base_state["dataset_id"] = "ds-test"
    base_state["target_column"] = "churned"
    base_state["task_type"] = "classification"
    base_state["selected_model"] = "logistic_regression"
    base_state["tuning_performed"] = False
    base_state["tuning_reason"] = "already strong"
    base_state["saved_model_info"] = {"saved_model_path": "/some/path.joblib"}
    base_state["tool_results"] = [
        {
            "stage": "train",
            "result": {
                "primary_metric": "f1",
                "primary_metric_reason": "balances precision/recall",
                "task_type": "classification",
                "results": [
                    {
                        "model": "logistic_regression",
                        "cv_mean": {"f1": 0.8821},
                        "cv_std": {"f1": 0.02},
                        "test_metrics": {"f1": 0.87, "accuracy": 0.9},
                        "diagnostic": {"diagnostic": "no_strong_signal", "evidence": "train=0.9, val=0.88"},
                    }
                ],
            },
        }
    ]
    mock_llm([FinalNarrative(comparison_summary="Logistic regression performed best.", caveats=["Small dataset."])])

    update = finalize_node(base_state)
    response = update["final_response"]

    assert response.success is True
    assert response.selected_model == "logistic_regression"
    assert response.primary_metric == "f1"
    # the exact cv_mean value from tool_results must appear in the grounded text
    assert "0.8821" in response.validation_performance
    assert response.saved_model_location == "/some/path.joblib"
    assert response.comparison_summary == "Logistic regression performed best."
    assert response.caveats == ["Small dataset."]
