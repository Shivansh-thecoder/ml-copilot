"""The ML Copilot orchestration graph.

    START
      |
      v
    planner                      (structured plan, via with_structured_output())
      |
      v
    inspect ------------------- required: failure -> finalize
      |
      v
    profile ------------------- required: failure -> finalize
      |
      v
    detect_target -------------- required: failure -> finalize
      |
      +-- needs_clarification --> clarification --+
      |                             (interrupt())  |
      |    <-------------------------------------- +   (loops back, bounded by
      |                                                  MAX_ITERATIONS -- see
      v                                                  copilot_control.py)
    train ---------------------- required: failure -> finalize
      |
      v
    tune_decision                (pure Python -- no LLM, no MCP call)
      |
      +-- skip --------------------------+
      |                                  |
      v                                  |
    tune ----- optional: failure is -----+   (falls back to the un-tuned
      |         recoverable, continues       baseline; noted as a caveat)
      v
    save ------------------------------------ optional: failure is
      |                                        recoverable, continues
      v
    retrieve_results ------------------------- optional: failure is
      |                                        recoverable, continues
      v
    finalize                      (structured CopilotFinalResponse,
      |                            grounded in tool_results -- see
      v                            copilot_control.py)
    END

WHY THIS IS A BOUNDED DAG, NOT A ReAct-STYLE AGENT LOOP
------------------------------------------------------------
Every node above runs AT MOST ONCE per graph invocation, except
`clarification` <-> `detect_target`, which is the one legitimate cycle --
bounded by construction (each pass requires a real human response via
`interrupt()`, and `MAX_ITERATIONS`/`MAX_TOOL_CALLS`/
`MAX_CONSECUTIVE_FAILURES`, defined in nodes/copilot_common.py, cap it
regardless). There is no general "keep calling tools until the LLM
decides to stop" loop anywhere in this graph -- the SEQUENCE of stages is
fixed by these edges, not chosen by the model at each step. What the
model DOES choose, at each individual stage, is WHICH of the currently
MCP-discovered tools best matches that stage's instruction (see
nodes/copilot_tools.py's `run_tool_stage`) -- tool selection is dynamic;
workflow structure is not.

REQUIRED vs. OPTIONAL STAGES
---------------------------------
inspect/profile/detect_target/train are REQUIRED: the workflow cannot
produce a meaningful answer without them, so a failure (after
`copilot_tools.py`'s internal per-stage retries are exhausted) routes
straight to `finalize`, which reports why the workflow stopped rather
than fabricating a result. tune/save/retrieve_results are OPTIONAL: a
failure there is recoverable -- the workflow still finishes and reports
its results, noting the limitation, rather than discarding an otherwise-
successful training run over a failed save or an unavailable results
lookup.

CHECKPOINTING
-----------------
Compiled with `MemorySaver`. It's what makes `clarification_node`'s
`interrupt()`/resume work at all: the graph's state at the moment of
interruption is what gets persisted and restored.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from nodes.copilot_common import last_result
from nodes.copilot_control import clarification_node, finalize_node, tune_decision_node
from nodes.copilot_planner import planner_node
from nodes.copilot_tools import (
    detect_target_stage,
    inspect_stage,
    profile_stage,
    retrieve_results_stage,
    save_stage,
    train_stage,
    tune_stage,
)
from state import CopilotState


def _route_required(stage_name: str, next_node: str):
    """Routing for a REQUIRED stage: proceed on success, abort to
    `finalize` on failure. One factory instead of four near-identical
    functions -- each required stage differs only in its name and where
    success leads."""

    def _route(state: CopilotState) -> str:
        if last_result(state, stage_name) is None:
            return "finalize"
        return next_node

    _route.__name__ = f"route_after_{stage_name}"
    return _route


def route_after_detect_target(state: CopilotState) -> str:
    if last_result(state, "detect_target") is None:
        return "finalize"
    if state["needs_clarification"]:
        return "clarification"
    return "train"


def route_after_tune_decision(state: CopilotState) -> str:
    return "tune" if state["tuning_performed"] else "save"


def build_copilot_graph():
    """Construct and compile the Copilot graph, with checkpointing enabled."""
    builder = StateGraph(CopilotState)

    builder.add_node("planner", planner_node)
    builder.add_node("inspect", inspect_stage)
    builder.add_node("profile", profile_stage)
    builder.add_node("detect_target", detect_target_stage)
    builder.add_node("clarification", clarification_node)
    builder.add_node("train", train_stage)
    builder.add_node("tune_decision", tune_decision_node)
    builder.add_node("tune", tune_stage)
    builder.add_node("save", save_stage)
    builder.add_node("retrieve_results", retrieve_results_stage)
    builder.add_node("finalize", finalize_node)

    builder.add_edge(START, "planner")
    builder.add_edge("planner", "inspect")

    builder.add_conditional_edges(
        "inspect", _route_required("inspect", "profile"), {"profile": "profile", "finalize": "finalize"}
    )
    builder.add_conditional_edges(
        "profile", _route_required("profile", "detect_target"), {"detect_target": "detect_target", "finalize": "finalize"}
    )
    builder.add_conditional_edges(
        "detect_target",
        route_after_detect_target,
        {"clarification": "clarification", "train": "train", "finalize": "finalize"},
    )
    builder.add_edge("clarification", "detect_target")

    builder.add_conditional_edges(
        "train", _route_required("train", "tune_decision"), {"tune_decision": "tune_decision", "finalize": "finalize"}
    )
    builder.add_conditional_edges(
        "tune_decision", route_after_tune_decision, {"tune": "tune", "save": "save"}
    )

    # tune/save/retrieve_results are optional -- always proceed regardless
    # of failure (each stage already marks its own errors "recoverable").
    builder.add_edge("tune", "save")
    builder.add_edge("save", "retrieve_results")
    builder.add_edge("retrieve_results", "finalize")

    builder.add_edge("finalize", END)

    return builder.compile(checkpointer=MemorySaver())
