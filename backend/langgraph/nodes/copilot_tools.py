"""The MCP-calling stage nodes for the Copilot orchestration graph.

Each stage (`inspect_stage`, `profile_stage`, ...) is a thin wrapper
around `run_tool_stage()`: it builds a plain-English instruction from
values already in state (`dataset_id`, `target_column`, ...) describing
what needs to happen right now, and hands it to `run_tool_stage()`,
which:

    1. discovers what the MCP server currently exposes (`list_mcp_tools_sync()`)
    2. offers ALL of them to the model, forcing a tool call
       (`bind_tools(..., tool_choice="required")` -- never a plain-text non-answer)
    3. lets the model pick one, matching its description to the instruction
    4. executes whatever it picked, generically (`call_mcp_tool_sync(name, args)`)

No stage branches on a tool's name (`if tool_name == "inspect_dataset"`)
-- that would hardcode what should come from MCP discovery. What IS
fixed is the SEQUENCE of stages, via the graph's edges
(graphs/copilot_graph.py): "inspect before train" is a property of the
workflow, not of any tool's name.

`tool_choice="required"` plus taking only `response.tool_calls[0]` means
each `run_tool_stage()` call makes AT MOST one MCP tool invocation before
returning control to the graph -- the graph decides whether another stage
happens next, keeping the whole workflow a bounded DAG rather than an
open-ended agent loop.
"""

import json
import math

from llm.model import get_chat_model
from mcp_client.client import call_mcp_tool_sync, list_mcp_tools_sync
from nodes.copilot_common import MAX_RETRIES_PER_STAGE, last_result
from nodes.copilot_control import check_limits
from state import CopilotState

__all__ = [
    "run_tool_stage",
    "inspect_stage",
    "profile_stage",
    "detect_target_stage",
    "train_stage",
    "tune_stage",
    "save_stage",
    "retrieve_results_stage",
]


def _mcp_tool_to_openai_schema(tool) -> dict:
    """MCP's own name/description/inputSchema, converted to the tool-calling
    schema OpenAI's function-calling API expects -- no LangChain @tool involved."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description or "",
            "parameters": tool.inputSchema,
        },
    }


def run_tool_stage(state: CopilotState, stage_name: str, instruction: str) -> dict:
    """Ask the LLM to pick and call one MCP tool for this stage; return a
    LangGraph partial-state-update dict.

    On success: `tool_results` gains one entry, `completed_steps` gains
    `stage_name`, counters advance, `consecutive_failures` resets to 0.

    On failure (tool errored, or the model didn't call a tool, or its
    call raised) this retries up to `MAX_RETRIES_PER_STAGE` times, giving
    the model the previous error as context each retry. If still failing
    after that, returns an `errors` entry instead -- the graph's routing
    functions (graphs/copilot_graph.py) decide whether that's fatal for
    this particular stage (see that file's REQUIRED_STAGES).
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(
        content=(
            "You are the ML Copilot's orchestration assistant. You MUST call exactly one "
            "of the tools made available to you to carry out the following instruction. "
            "Pick the tool whose description best matches the instruction, and supply "
            "correct arguments based on the context given in the instruction."
        )
    )
    human = HumanMessage(content=instruction)

    mcp_tools = list_mcp_tools_sync()
    tool_schemas = [_mcp_tool_to_openai_schema(t) for t in mcp_tools]
    model = get_chat_model().bind_tools(tool_schemas, tool_choice="required")

    conversation = [system, human]
    last_error = None
    tool_calls_made = 0

    for _attempt in range(MAX_RETRIES_PER_STAGE):
        response = model.invoke(conversation)
        if not response.tool_calls:
            last_error = "The model did not call a tool as required."
            conversation.append(response)
            conversation.append(HumanMessage(content=f"{last_error} You must call a tool. Try again."))
            continue

        call = response.tool_calls[0]
        tool_calls_made += 1
        try:
            raw_result = call_mcp_tool_sync(call["name"], call["args"])
            parsed = json.loads(raw_result)
        except Exception as exc:  # MCP call failed, or result wasn't valid JSON
            last_error = f"Calling '{call['name']}' with {call['args']} failed: {exc}"
            conversation.append(response)
            conversation.append(HumanMessage(content=f"{last_error} Try again with corrected arguments."))
            continue

        if isinstance(parsed, dict) and parsed.get("error"):
            last_error = parsed.get("message", parsed["error"])
            conversation.append(response)
            conversation.append(HumanMessage(content=f"Tool reported an error: {last_error} Try again."))
            continue

        return {
            "tool_results": [{"stage": stage_name, "tool": call["name"], "args": call["args"], "result": parsed}],
            "completed_steps": [stage_name],
            "tool_call_count": state["tool_call_count"] + tool_calls_made,
            "iteration_count": state["iteration_count"] + 1,
            "consecutive_failures": 0,
        }

    return {
        "errors": [{"stage": stage_name, "message": last_error or "unknown failure", "recoverable": False}],
        "tool_call_count": state["tool_call_count"] + tool_calls_made,
        "iteration_count": state["iteration_count"] + 1,
        "consecutive_failures": state["consecutive_failures"] + 1,
    }


def inspect_stage(state: CopilotState) -> dict:
    """Stage: inspect_dataset -- structural facts before any other analysis."""
    limit_error = check_limits(state)
    if limit_error:
        return {"errors": [limit_error]}
    instruction = (
        f"Inspect the structure (shape, column types, missing values, duplicates) of "
        f"dataset '{state['dataset_id']}'. This must happen before any other analysis."
    )
    return run_tool_stage(state, "inspect", instruction)


def profile_stage(state: CopilotState) -> dict:
    """Stage: profile_dataset -- EDA (outliers, correlations, distributions)."""
    limit_error = check_limits(state)
    if limit_error:
        return {"errors": [limit_error]}
    instruction = (
        f"Run exploratory data analysis (outlier signals, correlations, distributions) "
        f"on dataset '{state['dataset_id']}'."
    )
    return run_tool_stage(state, "profile", instruction)


def detect_target_stage(state: CopilotState) -> dict:
    """Stage: detect_target -- resolve target column + task type, or flag ambiguity."""
    limit_error = check_limits(state)
    if limit_error:
        return {"errors": [limit_error]}

    target_hint = (
        f" The target column is already known to be '{state['target_column']}' -- use it directly, "
        f"do not re-guess."
        if state.get("target_column")
        else ""
    )
    instruction = (
        f"Determine the target column and ML task type (classification or regression) for "
        f"dataset '{state['dataset_id']}'.{target_hint}"
    )
    update = run_tool_stage(state, "detect_target", instruction)
    if "errors" in update:
        return update

    result = update["tool_results"][0]["result"]
    if result.get("status") == "needs_clarification":
        candidates = result.get("candidates") or []
        if candidates:
            candidate_list = ", ".join(f"'{c['column']}'" for c in candidates)
            question = (
                f"I couldn't confidently determine the target column for objective "
                f"'{state['objective']}'. Candidates I considered: {candidate_list}. "
                f"Which column should I predict?"
            )
        else:
            question = (
                f"I couldn't find a plausible target column in this dataset for objective "
                f"'{state['objective']}'. Which column should I predict?"
            )
        update["needs_clarification"] = True
        update["clarification_question"] = question
    else:
        update["target_column"] = result["target_column"]
        update["task_type"] = result["task_type"]
        update["needs_clarification"] = False
        update["clarification_question"] = None
    return update


def train_stage(state: CopilotState) -> dict:
    """Stage: train_models -- baselines + CV + test eval + diagnostics, in one call."""
    limit_error = check_limits(state)
    if limit_error:
        return {"errors": [limit_error]}
    instruction = (
        f"Train and compare baseline models for dataset '{state['dataset_id']}', "
        f"target column '{state['target_column']}', task type '{state['task_type']}'."
    )
    update = run_tool_stage(state, "train", instruction)
    if "errors" in update:
        return update

    result = update["tool_results"][0]["result"]
    update["run_id"] = result["run_id"]
    update["selected_model"] = result["best_model"]
    return update


def tune_stage(state: CopilotState) -> dict:
    """Stage: tune_model -- only reached if tune_decision_node set tuning_performed=True.

    A tuning FAILURE is treated as recoverable, not fatal: the workflow
    falls back to the already-trained baseline (run_id/selected_model
    from train_stage are left untouched) and continues to save_stage --
    see graphs/copilot_graph.py's routing, which never treats this
    stage's errors as blocking.
    """
    limit_error = check_limits(state)
    if limit_error:
        return {"errors": [limit_error]}
    instruction = (
        f"Run a bounded hyperparameter search for the '{state['selected_model']}' model on "
        f"dataset '{state['dataset_id']}', target column '{state['target_column']}', "
        f"task type '{state['task_type']}'."
    )
    train_result = last_result(state, "train")
    evaluation = train_result.get("evaluation_configuration") if train_result else None
    if evaluation:
        instruction += (
            f" Use cv_folds={evaluation['cv_folds']} and test_size={evaluation['test_size']} "
            "to match the baseline evaluation."
        )
    update = run_tool_stage(state, "tune", instruction)
    if "errors" in update:
        # Recoverable: mark it so, but don't let it look like the fatal
        # kind graphs/copilot_graph.py's routing checks for on required stages.
        update["errors"][0]["recoverable"] = True
        return update

    result = update["tool_results"][0]["result"]
    train_result = last_result(state, "train")
    baseline = next(r for r in train_result["results"] if r["model"] == state["selected_model"])
    metric = train_result["primary_metric"]
    baseline_score = baseline["cv_mean"][metric]
    tuned_score = result.get("best_cv_score")
    comparable = (
        result.get("primary_metric") == metric
        and result.get("model") == state["selected_model"]
        and result.get("evaluation_configuration") == train_result.get("evaluation_configuration")
        and isinstance(tuned_score, (int, float))
        and math.isfinite(tuned_score)
        and math.isfinite(baseline_score)
    )
    improved = comparable and (
        tuned_score < baseline_score if metric in {"rmse", "mae"} else tuned_score > baseline_score
    )
    if improved:
        update["run_id"] = result["run_id"]
        outcome = "Tuned model selected because its cross-validation score improved."
    else:
        outcome = "Baseline retained because tuning did not produce a comparable, strictly better cross-validation score."
    update["tuning_reason"] = f"{state.get('tuning_reason', '')} {outcome}".strip()
    return update


def save_stage(state: CopilotState) -> dict:
    """Stage: save_model -- persist the current best (possibly tuned) model.

    Save failure is recoverable -- the workflow still reports everything
    it learned, noting the model wasn't persisted, rather than discarding
    an otherwise-successful run's results.
    """
    limit_error = check_limits(state)
    if limit_error:
        return {"errors": [limit_error]}
    instruction = (
        f"Save the '{state['selected_model']}' model from run '{state['run_id']}' as the "
        f"saved model for dataset '{state['dataset_id']}'."
    )
    update = run_tool_stage(state, "save", instruction)
    if "errors" in update:
        update["errors"][0]["recoverable"] = True
        return update

    update["saved_model_info"] = update["tool_results"][0]["result"]
    return update


def retrieve_results_stage(state: CopilotState) -> dict:
    """Stage: get_experiment_results -- the evidence finalize_node grounds its answer in."""
    limit_error = check_limits(state)
    if limit_error:
        return {"errors": [limit_error]}
    instruction = f"Retrieve the experiment/run history and saved-model info for dataset '{state['dataset_id']}'."
    update = run_tool_stage(state, "retrieve_results", instruction)
    if "errors" in update:
        update["errors"][0]["recoverable"] = True
    return update
