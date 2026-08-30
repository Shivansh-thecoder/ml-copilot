from schemas.copilot import CopilotPlan


def test_planner_node_records_plan_without_fabricating_target(base_state, mock_llm):
    from nodes.copilot_planner import planner_node

    plan = CopilotPlan(
        objective="Build a model to predict customer churn.",
        dataset_id="ds-test",
        target_column=None,
        task_type=None,
        intended_analysis_steps=["Inspect the dataset", "Profile for EDA"],
        intended_modeling_steps=["Train baselines", "Evaluate", "Tune if useful"],
    )
    mock_llm([plan])

    update = planner_node(base_state)

    assert update["plan"] is plan
    assert "target_column" not in update  # never set when the plan didn't know it
    assert update["completed_steps"] == ["planner"]


def test_planner_node_uses_explicit_target_from_objective(base_state, mock_llm):
    from nodes.copilot_planner import planner_node

    plan = CopilotPlan(
        objective="Predict churn using the 'churned' column.",
        dataset_id="ds-test",
        target_column="churned",
        task_type="classification",
        intended_analysis_steps=["Inspect the dataset"],
        intended_modeling_steps=["Train baselines"],
    )
    mock_llm([plan])

    update = planner_node(base_state)

    assert update["target_column"] == "churned"
    assert update["task_type"] == "classification"
