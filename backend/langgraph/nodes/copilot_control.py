"""Control-flow nodes for the Copilot orchestration graph: limits, the
tuning decision, clarification (pause/resume), and the final response.

None of these nodes call an MCP tool -- that's `copilot_tools.py`'s job.
These are the "LangGraph: orchestration, state, routing, iteration
limits" and "LLM: reasoning, planning, interpretation, explanation"
responsibilities from the architecture boundary, kept out of the
tool-calling stages so each file has one clear kind of job.
"""

import json

from langchain_core.messages import HumanMessage
from langgraph.types import interrupt

from llm.model import get_chat_model
from nodes.copilot_common import (
    MAX_CONSECUTIVE_FAILURES,
    MAX_ITERATIONS,
    MAX_TOOL_CALLS,
    last_result,
)
from schemas.copilot import CopilotFinalResponse, FinalNarrative
from state import CopilotState


def check_limits(state: CopilotState) -> dict | None:
    """Return an error dict if any bound has been exceeded, else None.

    Called at the top of every MCP-calling stage (copilot_tools.py) --
    this is the "explicit limits... maximum graph iterations... maximum
    tool calls... maximum consecutive failures" requirement, enforced as
    one shared check rather than scattered per-node conditionals. In this
    graph's actual topology (a bounded DAG, see graphs/copilot_graph.py)
    none of these should ever trigger -- they're a safety net, not the
    primary termination mechanism, which is the DAG's shape itself.
    """
    if state["iteration_count"] >= MAX_ITERATIONS:
        return {"stage": "limits", "message": f"Maximum iterations ({MAX_ITERATIONS}) reached.", "recoverable": False}
    if state["tool_call_count"] >= MAX_TOOL_CALLS:
        return {"stage": "limits", "message": f"Maximum tool calls ({MAX_TOOL_CALLS}) reached.", "recoverable": False}
    if state["consecutive_failures"] >= MAX_CONSECUTIVE_FAILURES:
        return {
            "stage": "limits",
            "message": f"Maximum consecutive failures ({MAX_CONSECUTIVE_FAILURES}) reached.",
            "recoverable": False,
        }
    return None


def tune_decision_node(state: CopilotState) -> dict:
    """Decide whether tuning is worth running -- pure Python, no LLM, no MCP call.

    This is an ORCHESTRATION decision (routing logic), not ML
    computation -- it reads numbers `train_stage` already produced via
    the ML engine and applies a fixed rule; it never computes a metric
    itself. Rules:

      possible_overfitting   -> tune (a regularization-aware search may help)
      possible_underfitting  -> skip (tuning the SAME model family is unlikely
                                 to fix underfitting; a different model would be
                                 needed, which is out of this workflow's scope)
      no_strong_signal       -> tune only if there's room to improve (primary
                                 metric, or R² for regression since RMSE has no
                                 universal "strong" threshold, below 0.9)
    """
    train_result = last_result(state, "train")
    best = next(r for r in train_result["results"] if r["model"] == state["selected_model"])
    diagnostic = best["diagnostic"]["diagnostic"]

    if diagnostic == "possible_overfitting":
        decision = True
        reason = (
            f"The selected model ('{state['selected_model']}') showed signs of overfitting "
            f"({best['diagnostic']['evidence']}), so a bounded hyperparameter search was run to "
            f"look for better-regularized settings."
        )
    elif diagnostic == "possible_underfitting":
        decision = False
        reason = (
            f"The selected model showed signs of underfitting ({best['diagnostic']['evidence']}). "
            f"Tuning the same model's hyperparameters is unlikely to fix underfitting -- a "
            f"different model family or additional features would be needed, which is beyond "
            f"this workflow -- so tuning was skipped."
        )
    else:
        strength_score = best["cv_mean"].get("r2", 0.0) if train_result["task_type"] == "regression" else best["cv_mean"][train_result["primary_metric"]]
        strong = strength_score >= 0.9
        decision = not strong
        if strong:
            reason = (
                f"The selected model already performs strongly ({train_result['primary_metric']}="
                f"{best['cv_mean'][train_result['primary_metric']]:.4f} in cross-validation) with no "
                f"strong over/underfitting signal, so tuning was skipped as unlikely to yield "
                f"meaningful improvement for the added computation."
            )
        else:
            reason = (
                f"The selected model performs reasonably but has room to improve "
                f"({train_result['primary_metric']}={best['cv_mean'][train_result['primary_metric']]:.4f} "
                f"in cross-validation) with no strong over/underfitting signal, so a bounded "
                f"hyperparameter search was run."
            )

    return {"tuning_performed": decision, "tuning_reason": reason}


def clarification_node(state: CopilotState) -> dict:
    """Pause the graph and wait for the user to resolve an ambiguous target.

    `interrupt(payload)` (LangGraph's built-in human-in-the-loop
    primitive, not a custom memory framework) halts graph execution HERE,
    persisted by the graph's checkpointer (see
    graphs/copilot_graph.py's `MemorySaver`). The caller sees the
    interrupt payload and, once it has an answer, resumes with
    `graph.invoke(Command(resume=answer), config)` using the SAME
    thread_id -- execution continues from exactly this point, with
    `interrupt(...)` returning `answer` instead of pausing again. See
    `app/api/routers/copilot.py`'s `continue_run` for the caller side.
    """
    answer = interrupt(
        {
            "question": state["clarification_question"],
            "dataset_id": state["dataset_id"],
        }
    )
    return {
        "target_column": answer,
        "needs_clarification": False,
        "clarification_question": None,
        "messages": [HumanMessage(content=f"Use '{answer}' as the target column.")],
        # Counted like any other step: a human repeatedly giving an
        # invalid column name would otherwise let this node/detect_target
        # cycle indefinitely without ever touching MAX_TOOL_CALLS (this
        # node makes no tool call itself) -- incrementing here means
        # MAX_ITERATIONS still eventually bounds even that case.
        "iteration_count": state["iteration_count"] + 1,
    }


def finalize_node(state: CopilotState) -> dict:
    """Build the structured final response.

    Every factual/numeric field is copied directly from
    `state["tool_results"]` -- never regenerated by an LLM call. Only
    `comparison_summary`/`caveats` (via `FinalNarrative`) are model-
    authored, and that call is given the actual results as context
    specifically so it can't invent a number that isn't there.
    """
    train_result = last_result(state, "train")

    if train_result is None:
        # A REQUIRED stage never succeeded -- the workflow cannot report
        # results, only why it stopped.
        failure_reason = "; ".join(e["message"] for e in state["errors"]) or "Unknown failure."
        response = CopilotFinalResponse(
            success=False,
            objective=state["objective"],
            dataset_id=state["dataset_id"],
            target_column=state.get("target_column"),
            task_type=state.get("task_type"),
            selected_model=None,
            primary_metric=None,
            validation_performance="N/A -- the workflow did not reach model training.",
            comparison_summary="N/A",
            overfitting_diagnostic=None,
            tuning_performed=False,
            tuning_explanation="N/A",
            saved_model_location=None,
            caveats=[],
            failure_reason=failure_reason,
        )
        return {"final_response": response, "current_step": "done"}

    best = next(r for r in train_result["results"] if r["model"] == state["selected_model"])
    tune_result = last_result(state, "tune") if state.get("tuning_performed") else None

    if tune_result:
        primary_metric = tune_result["primary_metric"]
        validation_performance = (
            f"After tuning: best cross-validated {primary_metric}={tune_result['best_cv_score']:.4f}; "
            f"held-out test metrics: {_format_metrics(tune_result['test_metrics'])}."
        )
        overfitting_diagnostic = f"{tune_result['diagnostic']['diagnostic']} ({tune_result['diagnostic']['evidence']})"
    else:
        primary_metric = train_result["primary_metric"]
        validation_performance = (
            f"Cross-validated {primary_metric}={best['cv_mean'][primary_metric]:.4f} "
            f"(std={best['cv_std'][primary_metric]:.4f}); held-out test metrics: "
            f"{_format_metrics(best['test_metrics'])}."
        )
        overfitting_diagnostic = f"{best['diagnostic']['diagnostic']} ({best['diagnostic']['evidence']})"

    saved_info = state.get("saved_model_info")
    saved_location = saved_info["saved_model_path"] if saved_info else None

    context = json.dumps(
        {
            "objective": state["objective"],
            "train_comparison": [
                {
                    "model": r["model"],
                    "cv_mean": r["cv_mean"],
                    "test_metrics": r["test_metrics"],
                    "diagnostic": r["diagnostic"]["diagnostic"],
                }
                for r in train_result["results"]
            ],
            "primary_metric": train_result["primary_metric"],
            "primary_metric_reason": train_result["primary_metric_reason"],
            "selected_model": state["selected_model"],
            "tuning_performed": state["tuning_performed"],
            "tuning_reason": state.get("tuning_reason", ""),
            "tune_result": tune_result,
            "model_saved": saved_info is not None,
            "recoverable_issues": [e["message"] for e in state["errors"] if e.get("recoverable")],
        },
        default=str,
    )

    narrative_model = get_chat_model().with_structured_output(FinalNarrative)
    narrative = narrative_model.invoke(
        "Given these ACTUAL ML results for an ML Copilot run (do not invent any number not "
        f"present here), write a plain-English comparison summary and a list of caveats for "
        f"the user.\n\nResults:\n{context}"
    )

    response = CopilotFinalResponse(
        success=True,
        objective=state["objective"],
        dataset_id=state["dataset_id"],
        target_column=state["target_column"],
        task_type=state["task_type"],
        selected_model=state["selected_model"],
        primary_metric=primary_metric,
        validation_performance=validation_performance,
        comparison_summary=narrative.comparison_summary,
        overfitting_diagnostic=overfitting_diagnostic,
        tuning_performed=state["tuning_performed"],
        tuning_explanation=state.get("tuning_reason", ""),
        saved_model_location=saved_location,
        caveats=narrative.caveats,
        failure_reason=None,
    )
    return {"final_response": response, "current_step": "done"}


def _format_metrics(metrics: dict) -> str:
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())
