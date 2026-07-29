import re
from typing import Protocol

from app.errors import ErrorCode, NonRetryableToolError, RetryableToolError
from app.state import MAX_RETRIES, AgentState, NextAction

# intent -> tool name. Extend as new tools land in app/tools.
READ_TOOLS = {
    "project_status": "get_project_status",
    "list_risks": "list_risks",
}
WRITE_TOOLS = {
    "create_risk": "create_risk",
}

_PROJECT_CODE_RE = re.compile(r"[A-Z]{2,}-\d+")


class ToolFn(Protocol):
    def __call__(self, tool_input: dict) -> dict: ...


class ToolRegistry(Protocol):
    """Contract app/tools must satisfy: look up a tool function by name."""

    def get(self, tool_name: str) -> ToolFn | None: ...


class IntentClassifier(Protocol):
    def __call__(self, message: str) -> tuple[str, dict]: ...


def default_classify(message: str) -> tuple[str, dict]:
    """Deterministic, offline keyword classifier.

    Placeholder for parse_intent until an LLMProvider-backed classifier is
    wired in; keeps the runtime demoable without an external dependency.
    """
    lowered = message.lower()
    match = _PROJECT_CODE_RE.search(message.upper())
    project_code = match.group(0) if match else None

    if "risk" in lowered and any(w in lowered for w in ("create", "add", "log", "new")):
        intent = "create_risk"
    elif "risk" in lowered:
        intent = "list_risks"
    elif "status" in lowered:
        intent = "project_status"
    else:
        intent = "unsupported"

    tool_input = {"project_code": project_code} if project_code else {}
    return intent, tool_input


def _evolve(state: AgentState, **updates) -> AgentState:
    """Build the next state through the constructor (not model_copy) so
    AgentState's validators actually run on every transition."""
    data = state.model_dump()
    data.update(updates)
    return AgentState(**data)


def parse_intent(message: str, classify: IntentClassifier = default_classify) -> AgentState:
    intent, tool_input = classify(message)
    return AgentState(
        intent=intent,
        tool_input=tool_input or None,
        next_action=NextAction.ROUTE_DECISION,
    )


def route_decision(state: AgentState) -> AgentState:
    if state.intent in READ_TOOLS:
        return _evolve(
            state,
            selected_tool=READ_TOOLS[state.intent],
            next_action=NextAction.EXECUTE_READ_TOOL,
        )
    if state.intent in WRITE_TOOLS:
        return _evolve(
            state,
            selected_tool=WRITE_TOOLS[state.intent],
            approval_required=True,
            next_action=NextAction.EXECUTE_WRITE_TOOL,
        )
    return _evolve(
        state,
        next_action=NextAction.FORMAT_RESPONSE,
        error_code=ErrorCode.UNSUPPORTED_INTENT,
        error_message=f"'{state.intent}' is not a supported request.",
    )


def _run_tool_with_retry(state: AgentState, tool_registry: ToolRegistry) -> AgentState:
    tool_fn = tool_registry.get(state.selected_tool)
    if tool_fn is None:
        return _evolve(
            state,
            next_action=NextAction.FORMAT_RESPONSE,
            error_code=ErrorCode.NOT_FOUND,
            error_message=f"No tool registered for '{state.selected_tool}'.",
        )

    retry_count = state.retry_count
    while True:
        try:
            output = tool_fn(state.tool_input or {})
            return _evolve(
                state,
                tool_output=output,
                next_action=NextAction.FORMAT_RESPONSE,
                error_code=None,
                error_message=None,
                retry_count=retry_count,
            )
        except NonRetryableToolError as exc:
            return _evolve(
                state,
                next_action=NextAction.FORMAT_RESPONSE,
                error_code=exc.code,
                error_message=exc.message,
                retry_count=retry_count,
            )
        except RetryableToolError as exc:
            if retry_count >= MAX_RETRIES:
                return _evolve(
                    state,
                    next_action=NextAction.FORMAT_RESPONSE,
                    error_code=exc.code,
                    error_message=exc.message,
                    retry_count=retry_count,
                )
            retry_count += 1


def execute_read_tool(state: AgentState, tool_registry: ToolRegistry) -> AgentState:
    return _run_tool_with_retry(state, tool_registry)


def execute_write_tool(state: AgentState, tool_registry: ToolRegistry) -> AgentState:
    if state.approved is None:
        return _evolve(state, next_action=NextAction.AWAIT_APPROVAL)
    if state.approved is False:
        return _evolve(
            state,
            next_action=NextAction.FORMAT_RESPONSE,
            error_code=ErrorCode.APPROVAL_REJECTED,
            error_message="The requested action was rejected during approval.",
        )
    return _run_tool_with_retry(state, tool_registry)


def _render_tool_output(tool_name: str | None, output: dict) -> str:
    # Placeholder rendering; refine once app/tools output shapes are final.
    return f"{tool_name}: {output}"


def format_response(state: AgentState) -> AgentState:
    if state.next_action == NextAction.AWAIT_APPROVAL:
        return state

    if state.error_code is not None:
        answer = state.error_message or "The request could not be completed."
    elif state.tool_output is not None:
        answer = _render_tool_output(state.selected_tool, state.tool_output)
    else:
        answer = "No result available."

    return _evolve(state, answer=answer, next_action=NextAction.DONE)


def run(
    message: str,
    tool_registry: ToolRegistry,
    classify: IntentClassifier = default_classify,
) -> AgentState:
    """Drive a request from raw message through to a final or
    approval-pending AgentState."""
    state = parse_intent(message, classify)
    state = route_decision(state)

    if state.next_action == NextAction.EXECUTE_READ_TOOL:
        state = execute_read_tool(state, tool_registry)
    elif state.next_action == NextAction.EXECUTE_WRITE_TOOL:
        state = execute_write_tool(state, tool_registry)

    if state.next_action == NextAction.AWAIT_APPROVAL:
        return state

    return format_response(state)


def resume(state: AgentState, tool_registry: ToolRegistry) -> AgentState:
    """Continue a write-tool flow after /approve or /reject has set
    `state.approved`."""
    if state.next_action != NextAction.AWAIT_APPROVAL:
        raise ValueError(
            f"resume() called on a state that is not awaiting approval: {state.next_action}"
        )
    state = execute_write_tool(state, tool_registry)
    return format_response(state)
