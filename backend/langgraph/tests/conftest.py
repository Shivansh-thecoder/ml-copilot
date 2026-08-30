"""Shared fixtures for the Copilot orchestration test suite.

`sys.path` gets `backend/langgraph` (this file's parent) inserted -- the
same directory Python puts on `sys.path[0]` automatically when the
package's own modules run -- so `import state`, `import nodes...` work
correctly here. Deliberately `backend/langgraph`, not `backend`: see
pytest.ini's comment -- inserting `backend` itself would let `import
langgraph` resolve to this directory instead of the installed SDK (they
share a name).
"""

import sys
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class FakeTool:
    """Stand-in for `mcp.types.Tool` -- just the three fields
    `nodes/copilot_tools.py`'s `_mcp_tool_to_openai_schema` reads."""

    def __init__(self, name: str, description: str = "", input_schema: dict | None = None):
        self.name = name
        self.description = description
        self.inputSchema = input_schema or {"type": "object", "properties": {}}


MOCK_MCP_TOOLS = [
    FakeTool("inspect_dataset", "Inspect a dataset's structure."),
    FakeTool("profile_dataset", "Run exploratory data analysis on a dataset."),
    FakeTool("detect_target", "Determine the target column and task type."),
    FakeTool("train_models", "Train and compare baseline models."),
    FakeTool("tune_model", "Run a bounded hyperparameter search."),
    FakeTool("save_model", "Save a trained model."),
    FakeTool("get_experiment_results", "Retrieve experiment history."),
]


class ScriptedModel:
    """One fake chat model serving a graph run's whole LLM call sequence.

    `.bind_tools(...).invoke(messages)` (tool-calling stages) and
    `.with_structured_output(Schema).invoke(prompt)` (planner/finalize)
    are deliberately served from TWO SEPARATE queues, not one shared
    ordered list. Reason: the number of tool-calling stages that actually
    run is data-dependent (tuning may or may not happen -- see
    `tune_decision_node`), while the two structured-output calls
    (planner, then finalize's narrative) always happen exactly once
    each, first and last. A single shared queue would silently
    mis-assign responses the moment the actual tool-call count differs
    from what a test assumed; two queues make that impossible by
    construction. `responses` (the constructor arg) still accepts one
    flat list for convenience in simple tests -- see `_install` below.
    """

    def __init__(self, tool_responses: list, structured_responses: list):
        self._tool_responses = list(tool_responses)
        self._tool_index = 0
        self._structured_responses = list(structured_responses)
        self._structured_index = 0
        self._mode = None  # set by whichever builder method was called last

    def bind_tools(self, tools, tool_choice=None):
        self._mode = "tool"
        return self

    def with_structured_output(self, schema):
        self._mode = "structured"
        return self

    def invoke(self, arg):
        queue, index_attr = (
            (self._tool_responses, "_tool_index") if self._mode == "tool" else (self._structured_responses, "_structured_index")
        )
        index = getattr(self, index_attr)
        if index >= len(queue):
            raise AssertionError(
                f"ScriptedModel ran out of '{self._mode}' responses after {index} calls -- "
                "the graph made more such LLM calls than the test scripted."
            )
        response = queue[index]
        setattr(self, index_attr, index + 1)
        if callable(response) and not isinstance(response, Exception):
            response = response(arg)
        if isinstance(response, Exception):
            raise response
        return response


def tool_call_message(name: str, args: dict, call_id: str = "call_1") -> AIMessage:
    """An AIMessage shaped like what `bind_tools(...).invoke()` returns
    when the model calls one tool -- what every tool-calling stage
    expects from `ScriptedModel`."""
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id}])


def no_tool_call_message(content: str = "I have no tool to call.") -> AIMessage:
    """An AIMessage with no tool_calls -- used to test the "model didn't
    call a tool" retry path in run_tool_stage."""
    return AIMessage(content=content, tool_calls=[])


@pytest.fixture
def mock_mcp_tools(monkeypatch):
    """Patch `list_mcp_tools_sync` (as imported into nodes.copilot_tools)
    to return the fixed MOCK_MCP_TOOLS list -- no real MCP subprocess."""
    import nodes.copilot_tools as copilot_tools

    monkeypatch.setattr(copilot_tools, "list_mcp_tools_sync", lambda: MOCK_MCP_TOOLS)
    return MOCK_MCP_TOOLS


@pytest.fixture
def mock_mcp_call(monkeypatch):
    """Patch `call_mcp_tool_sync` (as imported into nodes.copilot_tools)
    with a fake that returns pre-scripted JSON per tool name, tracking
    every call made for assertions.

    `.script(name, result)` registers ONE response, returned every time
    that tool is called. `.script(name, [result1, result2])` registers a
    SEQUENCE -- each call to that tool pops the next one (repeating the
    last once exhausted) -- needed when the same tool is legitimately
    called more than once with different expected results in one test
    (e.g. `detect_target` before and after a clarification answer).
    `result` (or any item in a sequence) may be an Exception, raised
    instead of returned.
    """
    import nodes.copilot_tools as copilot_tools

    calls = []
    scripts: dict[str, list] = {}

    def fake_call(name, args):
        calls.append((name, args))
        if name not in scripts:
            raise AssertionError(f"mock_mcp_call: no script registered for tool '{name}'")
        queue = scripts[name]
        result = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(result, Exception):
            raise result
        import json

        return json.dumps(result)

    def script(name, result):
        scripts[name] = list(result) if isinstance(result, list) else [result]

    fake_call.calls = calls
    fake_call.script = script
    monkeypatch.setattr(copilot_tools, "call_mcp_tool_sync", fake_call)
    return fake_call


@pytest.fixture
def mock_llm(monkeypatch):
    """Patch `get_chat_model` in every module that imports it, all
    returning the SAME `ScriptedModel` instance, so a single ordered
    response queue serves the whole graph run.

    Usage: `model = mock_llm([response1, response2, ...])`
    """
    import nodes.copilot_control as copilot_control
    import nodes.copilot_planner as copilot_planner
    import nodes.copilot_tools as copilot_tools
    from schemas.copilot import CopilotPlan, FinalNarrative

    def _install(responses: list) -> ScriptedModel:
        """Convenience form: one flat list, auto-split into the tool/
        structured queues by type -- a `CopilotPlan`/`FinalNarrative`
        instance goes to the structured queue (in the order given;
        planner's plan is always scripted before finalize's narrative),
        everything else (AIMessages, callables, Exceptions) goes to the
        tool queue, in order. Covers every test in this suite; call
        `ScriptedModel(tool_responses, structured_responses)` directly
        instead if a test genuinely needs to interleave them differently.
        """
        structured = [r for r in responses if isinstance(r, (CopilotPlan, FinalNarrative))]
        tools = [r for r in responses if not isinstance(r, (CopilotPlan, FinalNarrative))]
        model = ScriptedModel(tool_responses=tools, structured_responses=structured)
        monkeypatch.setattr(copilot_tools, "get_chat_model", lambda: model)
        monkeypatch.setattr(copilot_control, "get_chat_model", lambda: model)
        monkeypatch.setattr(copilot_planner, "get_chat_model", lambda: model)
        return model

    return _install


@pytest.fixture
def base_state():
    """A minimal, valid starting CopilotState -- individual tests
    override/extend fields as needed."""
    return {
        "objective": "Build a model to predict customer churn.",
        "dataset_id": "ds-test",
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
