"""Tiny helpers shared between `copilot_tools.py` and `copilot_control.py`.

Its own module specifically to avoid a circular import: `copilot_tools`
needs `check_limits` (defined in `copilot_control`), and
`copilot_control`'s `tune_decision_node`/`finalize_node` need
`last_result` (previously defined in `copilot_tools`) -- putting shared
pieces here breaks that cycle.
"""

from state import CopilotState

# Bounds for the whole orchestration run -- see graphs/copilot_graph.py's
# module docstring for why this graph's DAG shape already makes runaway
# looping structurally impossible; these are a safety net on top of that,
# not the primary termination mechanism.
MAX_ITERATIONS = 15
MAX_TOOL_CALLS = 20
MAX_CONSECUTIVE_FAILURES = 3

# Retries within a single stage's own tool-call attempt (copilot_tools.py).
MAX_RETRIES_PER_STAGE = 2


def last_result(state: CopilotState, stage_name: str) -> dict | None:
    """The most recent tool_results entry's `result` dict for a given
    stage, or None if that stage hasn't produced one (e.g. it failed, or
    hasn't run yet)."""
    for entry in reversed(state["tool_results"]):
        if entry["stage"] == stage_name:
            return entry["result"]
    return None
