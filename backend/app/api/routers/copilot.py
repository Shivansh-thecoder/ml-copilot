"""HTTP surface for the ML Copilot orchestration graph.

Three endpoints: start a run, continue one after a clarification pause,
retrieve its current result. No run-history list, cancellation, or
streaming/progress endpoint yet.

WHY THIS FILE IMPORTS FROM `backend/langgraph/`
-----------------------------------------------
Most files under `backend/app/` stay import-independent of
`backend/langgraph/` (see `langgraph/llm/model.py`). Routers whose whole
job is to expose something on the other side of the MCP boundary are the
exception: this one imports `graphs.copilot_graph`. Both packages already
share `backend/.venv`. The dependency only goes this one direction --
`backend/langgraph/` imports nothing from `backend/app/`.

SYNCHRONOUS, BLOCKING, IN-MEMORY
-------------------------------
`graph.invoke(...)` runs inside the request handler and blocks until the
graph pauses for clarification or finishes -- a real run takes tens of
seconds (multiple LLM calls, real model training). `MemorySaver` (the
graph's own checkpointer) keeps run state in this process's memory only,
so it is lost on restart. A production version would move the run to a
background task with a polling/streaming status endpoint.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.api.langgraph_bridge import langgraph_imports

# Imported through `langgraph_imports()` because `backend/langgraph/`
# shadows the installed `langgraph` SDK while `backend/` is on sys.path
# -- see app/api/langgraph_bridge.py for the full explanation.
with langgraph_imports():
    from graphs.copilot_graph import build_copilot_graph
    from langgraph.types import Command

router = APIRouter(prefix="/copilot", tags=["copilot"])

_graph = None


def _get_graph():
    """Lazy singleton -- one compiled graph (and its MemorySaver) reused
    across requests, rather than recompiled per call."""
    global _graph
    if _graph is None:
        _graph = build_copilot_graph()
    return _graph


def _initial_state(dataset_id: str, objective: str) -> dict:
    return {
        "objective": objective,
        "dataset_id": dataset_id,
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


def _summarize(run_id: str, result: dict) -> dict:
    if "__interrupt__" in result:
        return {
            "run_id": run_id,
            "status": "needs_clarification",
            "question": result["__interrupt__"][0].value["question"],
            "result": None,
        }
    response = result.get("final_response")
    return {
        "run_id": run_id,
        "status": "completed" if response else "in_progress",
        "question": None,
        "result": response.model_dump() if response else None,
    }


class StartRunRequest(BaseModel):
    dataset_id: str
    objective: str


class ContinueRunRequest(BaseModel):
    answer: str


@router.post("/runs")
def start_run(data: StartRunRequest) -> dict:
    """Start a new ML Copilot run for a dataset + objective. Blocks until
    the workflow either pauses for clarification or completes."""
    import uuid

    run_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": run_id}}
    result = _get_graph().invoke(_initial_state(data.dataset_id, data.objective), config)
    return _summarize(run_id, result)


@router.post("/runs/{run_id}/continue")
def continue_run(run_id: str, data: ContinueRunRequest) -> dict:
    """Resume a run that's paused for clarification, with the user's answer."""
    config = {"configurable": {"thread_id": run_id}}
    state = _get_graph().get_state(config)
    if not state.next:
        raise HTTPException(status_code=409, detail=f"Run '{run_id}' is not waiting for input.")
    result = _get_graph().invoke(Command(resume=data.answer), config)
    return _summarize(run_id, result)


@router.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Retrieve a run's current status/result without advancing it."""
    config = {"configurable": {"thread_id": run_id}}
    snapshot = _get_graph().get_state(config)
    if not snapshot.values:
        raise HTTPException(status_code=404, detail=f"No run found with id '{run_id}'.")
    values = dict(snapshot.values)
    if snapshot.next:  # graph is paused, waiting for input -- snapshot.interrupts
        values["__interrupt__"] = list(snapshot.interrupts)  # holds the pending Interrupt(s) directly
    return _summarize(run_id, values)
