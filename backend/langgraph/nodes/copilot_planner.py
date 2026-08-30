"""The planner node -- produces a structured plan via `with_structured_output()`.

The ONLY node in this graph that runs before the dataset has been
touched at all. Its `CopilotPlan` output records the objective and
whatever the user's own words state explicitly -- it must NOT guess a
target column or task type from the dataset, because it hasn't seen the
dataset; `detect_target_stage` (nodes/copilot_tools.py) is what actually
resolves those, grounded in real data.
"""

from llm.model import get_chat_model
from schemas.copilot import CopilotPlan
from state import CopilotState


def planner_node(state: CopilotState) -> dict:
    model = get_chat_model().with_structured_output(CopilotPlan)

    plan = model.invoke(
        "A user has given this objective for an ML Copilot workflow on dataset "
        f"'{state['dataset_id']}': \"{state['objective']}\"\n\n"
        "Produce a structured plan. Only set target_column/task_type if the objective "
        "text itself explicitly names them -- otherwise leave them null; they will be "
        "determined later by actually inspecting the dataset, not guessed now."
    )

    update = {
        "plan": plan,
        "current_step": "planned",
        "completed_steps": ["planner"],
        "iteration_count": state["iteration_count"] + 1,
    }
    if plan.target_column:
        update["target_column"] = plan.target_column
    if plan.task_type:
        update["task_type"] = plan.task_type
    return update
