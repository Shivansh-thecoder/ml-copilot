"""The LangGraph state for the ML Copilot orchestration graph.

State is the one object every node in the graph reads and writes. Each
node returns a partial update (just the keys it changed); LangGraph
merges that into the shared state before the next node runs. A field is
plain (overwritten on each write) unless annotated with a reducer, in
which case LangGraph combines the new value with the existing one
instead of replacing it -- see the field comments below for which fields
use `operator.add` and why.
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage

from schemas.copilot import CopilotFinalResponse, CopilotPlan


class CopilotState(TypedDict):
    """State for the bounded, multi-stage ML Copilot workflow:
    plan -> inspect -> profile -> detect target -> [clarify] -> train ->
    tune-or-not -> save -> retrieve results -> finalize.

    Grouped by role so the shape of the workflow is visible from the
    field list alone:

      - INPUT (set once, by the caller): objective, dataset_id
      - PLANNING (written once, by planner_node): plan
      - RESOLVED FACTS (written by detect_target_stage, possibly after
        a clarification round-trip): target_column, task_type
      - CONVERSATION: messages
      - EVIDENCE (one entry per MCP tool call, in order -- what
        finalize_node grounds its answer in): tool_results
      - PROGRESS TRACKING: current_step, completed_steps, errors
      - CLARIFICATION (interrupt/resume, see nodes/copilot_control.py):
        needs_clarification, clarification_question
      - MODELING DECISIONS (written by train/tune-decision/tune/save):
        run_id, selected_model, tuning_performed, tuning_reason,
        saved_model_info
      - OUTPUT (written once, by finalize_node): final_response
      - SAFETY / BOUNDS (see nodes/copilot_common.py's constants and
        check_limits()): iteration_count, tool_call_count,
        consecutive_failures
    """

    # --- input ---
    objective: str
    dataset_id: str

    # --- planning ---
    plan: CopilotPlan | None

    # --- resolved facts ---
    target_column: str | None
    task_type: str | None

    # --- conversation ---
    messages: Annotated[list[BaseMessage], operator.add]

    # --- evidence ---
    # operator.add: multiple stages append here over one run, and
    # finalize_node needs the full history, not just the last entry.
    tool_results: Annotated[list[dict], operator.add]  # [{"stage":..., "tool":..., "args":..., "result":...}]

    # --- progress tracking ---
    current_step: str
    completed_steps: Annotated[list[str], operator.add]
    errors: Annotated[list[dict], operator.add]  # [{"stage":..., "message":..., "recoverable": bool}]

    # --- clarification ---
    needs_clarification: bool
    clarification_question: str | None

    # --- modeling decisions ---
    run_id: str | None
    selected_model: str | None
    tuning_performed: bool
    tuning_reason: str
    saved_model_info: dict | None

    # --- output ---
    final_response: CopilotFinalResponse | None

    # --- safety / bounds ---
    iteration_count: int
    tool_call_count: int
    consecutive_failures: int
