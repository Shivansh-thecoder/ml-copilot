import pytest

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


@pytest.mark.parametrize('metric,baseline_score,tuned_score,accepted', [
    ('f1', .8, .7, False), ('f1', .8, .8, False), ('f1', .8, .9, True),
    ('rmse', 2., 3., False), ('rmse', 2., 2., False), ('rmse', 2., 1., True),
    ('f1', .8, float('nan'), False), ('f1', .8, float('inf'), False),
    ('f1', .80001, .80002, True),
])
def test_tuning_selection_and_final_report(metric, baseline_score, tuned_score, accepted,
                                           base_state, mock_llm, mock_mcp_tools, mock_mcp_call):
    from nodes.copilot_control import finalize_node
    from schemas.copilot import FinalNarrative

    base_state.update(selected_model='logistic_regression', run_id='baseline',
                      tuning_performed=True, task_type='classification', target_column='label')
    diagnostic = {'diagnostic': 'no_strong_signal', 'evidence': 'test evidence'}
    base_state['tool_results'] = [{'stage': 'train', 'result': {
        'primary_metric': metric, 'primary_metric_reason': 'test',
        'results': [{'model': 'logistic_regression', 'cv_mean': {metric: baseline_score},
                     'cv_std': {metric: .01}, 'test_metrics': {metric: .1234}, 'diagnostic': diagnostic}],
    }}]
    mock_mcp_call.script('tune_model', {
        'run_id': 'tuned', 'model': 'logistic_regression', 'primary_metric': metric,
        'best_cv_score': tuned_score, 'test_metrics': {metric: .9876}, 'diagnostic': diagnostic,
    })
    mock_llm([tool_call_message('tune_model', {}), FinalNarrative(comparison_summary='Test', caveats=[])])
    update = tune_stage(base_state)
    assert update.get('run_id', 'baseline') == ('tuned' if accepted else 'baseline')
    base_state['tool_results'].extend(update.pop('tool_results'))
    base_state.update(update)
    response = finalize_node(base_state)['final_response']
    assert ('After tuning' in response.validation_performance) == accepted
    assert ('.9876' if accepted else '.1234') in response.validation_performance
    assert response.tuning_performed is True
    if not accepted:
        assert 'Baseline retained' in response.tuning_explanation


def test_tuning_with_different_folds_keeps_baseline(base_state, monkeypatch):
    import nodes.copilot_tools as tools

    base_state.update(selected_model='logistic_regression', run_id='baseline')
    base_state['tool_results'] = [{'stage': 'train', 'result': {
        'primary_metric': 'f1',
        'evaluation_configuration': {'cv_folds': 5, 'test_size': .2},
        'results': [{'model': 'logistic_regression', 'cv_mean': {'f1': .8}}],
    }}]
    instructions = []

    def run_stage(state, stage, instruction):
        instructions.append(instruction)
        if stage == 'save':
            return {'tool_results': [{'result': {'saved_model_path': 'baseline.joblib'}}]}
        return {'tool_results': [{'stage': 'tune', 'result': {
            'model': 'logistic_regression', 'primary_metric': 'f1',
            'run_id': 'tuned', 'best_cv_score': .99,
            'evaluation_configuration': {'cv_folds': 2, 'test_size': .2},
        }}]}

    monkeypatch.setattr(tools, 'run_tool_stage', run_stage)
    update = tune_stage(base_state)
    assert 'run_id' not in update
    assert 'cv_folds=5' in instructions[0]
    base_state.update(update)
    save_stage(base_state)
    assert "from run 'baseline'" in instructions[-1]
