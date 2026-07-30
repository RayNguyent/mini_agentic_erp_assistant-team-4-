import re
from typing import TYPE_CHECKING, Protocol

from app.errors import ErrorCode, NonRetryableToolError, RetryableToolError
from app.state import MAX_RETRIES, AgentState, NextAction

if TYPE_CHECKING:
    from app.providers.base import LLMProvider

# intent -> tool name. Extend as new tools land in app/tools.
READ_TOOLS = {
    "project_status": "get_project_status",
    "list_risks": "list_risks",
}
WRITE_TOOLS = {
    "create_risk": "create_risk",
}

_PROJECT_CODE_RE = re.compile(r"[A-Za-z]{2,}[\s_-]?\d+")
_PROJECT_CODE_PARTS_RE = re.compile(r"([A-Za-z]{2,})[\s_-]?(\d+)")


def _normalize_project_code(code: str) -> str:
    """Canonicalize any extracted project code to LETTERS-DIGITS form (e.g.
    "prj 001", "PRJ_001", "PRJ001" -> "PRJ-001"), regardless of whether the
    deterministic regex or an LLM classifier produced it — tool execution
    always looks projects up by this exact canonical key (app/tools/erp_tools.py).

    Also handles variations like "project 2" -> "PRJ-002", "prj-2" -> "PRJ-002".
    """
    match = _PROJECT_CODE_PARTS_RE.match(code.strip())
    if not match:
        return code
    letters, digits = match.groups()

    # Normalize "project" or "proj" variations to "PRJ"
    letters_upper = letters.upper()
    if letters_upper.startswith("PROJECT"):
        letters_upper = "PRJ"
    elif letters_upper.startswith("PROJ"):
        letters_upper = "PRJ"

    # Pad digits to 3 digits if using PRJ prefix (e.g., "2" -> "002")
    if letters_upper == "PRJ" and len(digits) < 3:
        digits = digits.zfill(3)

    return f"{letters_upper}-{digits}"


class ToolFn(Protocol):
    def __call__(self, tool_input: dict) -> dict: ...


class ToolRegistry(Protocol):
    """Contract app/tools must satisfy: look up a tool function by name."""

    def get(self, tool_name: str) -> ToolFn | None: ...


class StatusCallback(Protocol):
    """Called with status updates during agent execution."""

    def __call__(self, status: str) -> None: ...


class IntentClassifier(Protocol):
    def __call__(self, message: str, history: list[dict] | None = None) -> tuple[str, dict]: ...


def default_classify(message: str, history: list[dict] | None = None) -> tuple[str, dict]:
    """Deterministic, offline keyword classifier.

    Placeholder for parse_intent until an LLMProvider-backed classifier is
    wired in; keeps the runtime demoable without an external dependency.
    `history` is accepted (to match IntentClassifier) but unused — this
    classifier has no way to reason over prior turns.
    """
    lowered = message.lower()
    match = _PROJECT_CODE_RE.search(message.upper())
    project_code = _normalize_project_code(match.group(0)) if match else None

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


def _infer_intent_from_history(history: list[dict] | None) -> str | None:
    """If the classifier declined to infer a follow-up from context, fall
    back to reusing the last read-tool intent from the conversation so a
    bare "how about PRJ-003?" still resolves. Deliberately excludes
    create_risk — a write action should never be inferred from an ambiguous
    follow-up, only from an explicit request."""
    if not history:
        return None
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        intent, _ = default_classify(turn.get("content", ""))
        if intent in READ_TOOLS:
            return intent
    return None


def _evolve(state: AgentState, **updates) -> AgentState:
    """Build the next state through the constructor (not model_copy) so
    AgentState's validators actually run on every transition."""
    data = state.model_dump()
    data.update(updates)
    return AgentState(**data)


def parse_intent(
    message: str,
    classify: IntentClassifier = default_classify,
    history: list[dict] | None = None,
) -> AgentState:
    intent, tool_input = classify(message, history)

    if intent == "unsupported":
        code_match = _PROJECT_CODE_RE.search(message.upper())
        carried_intent = _infer_intent_from_history(history) if code_match else None
        if carried_intent:
            intent = carried_intent
            tool_input = {"project_code": _normalize_project_code(code_match.group(0))}

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
            status="Executing tool...",
        )
    if state.intent in WRITE_TOOLS:
        return _evolve(
            state,
            selected_tool=WRITE_TOOLS[state.intent],
            approval_required=True,
            next_action=NextAction.EXECUTE_WRITE_TOOL,
            status="Requesting approval...",
        )
    return _evolve(
        state,
        next_action=NextAction.GENERATE_RESPONSE,
        status="Generating helpful response...",
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

    tool_input = state.tool_input or {}
    if isinstance(tool_input.get("project_code"), str):
        tool_input = {**tool_input, "project_code": _normalize_project_code(tool_input["project_code"])}

    retry_count = state.retry_count
    while True:
        try:
            output = tool_fn(tool_input)
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


def _render_project_status(output: dict) -> str:
    return (
        f"Project {output.get('project_code', '?')} — {output.get('name', '?')}\n"
        f"• Stage: {output.get('stage', '?')}\n"
        f"• Owner: {output.get('owner', '?')}\n"
        f"• Status: {output.get('status_summary', '?')}"
    )


def _render_risk(risk: dict) -> str:
    return f"• [{risk.get('id', '?')}] {risk.get('title', '?')} ({risk.get('severity', '?')}) — {risk.get('status', '?')}"


def _render_list_risks(output: dict) -> str:
    risks = output.get("risks", [])
    if not risks:
        return f"No risks recorded for {output.get('project_code', '?')}."
    header = f"Risks for {output.get('project_code', '?')}:"
    return "\n".join([header, *(_render_risk(risk) for risk in risks)])


def _render_create_risk(output: dict) -> str:
    return (
        f"Risk created for {output.get('project_code', '?')}:\n"
        f"• Title: {output.get('title', '?')}\n"
        f"• Severity: {output.get('severity', '?')}\n"
        f"• Status: {output.get('status', '?')}"
    )


_TOOL_RENDERERS = {
    "get_project_status": _render_project_status,
    "list_risks": _render_list_risks,
    "create_risk": _render_create_risk,
}


def _render_tool_output(tool_name: str | None, output: dict) -> str:
    renderer = _TOOL_RENDERERS.get(tool_name or "")
    if renderer is None:
        return f"{tool_name}: {output}"
    return renderer(output)


def generate_response(state: AgentState, message: str, provider: "LLMProvider | None" = None) -> AgentState:
    """Generate a conversational response when no tool applies."""
    response = generate_conversational_response(message, provider)
    return _evolve(
        state,
        answer=response,
        next_action=NextAction.DONE,
        status="Done!",
    )


def generate_conversational_response(message: str, provider: "LLMProvider | None" = None) -> str:
    """Generate a helpful conversational response when no tool applies.

    Falls back to a generic message if no provider is available."""
    if provider is None:
        return (
            "I'm an ERP assistant focused on project management. I can help you:\n"
            "• Check project status (e.g., 'What's the status of PRJ-001?')\n"
            "• List risks for a project (e.g., 'Show risks for PRJ-001')\n"
            "• Create new risks (e.g., 'Create a risk for PRJ-001')\n\n"
            "What can I help you with?"
        )

    system_prompt = """You are a helpful ERP project management assistant. The user has asked
something that doesn't directly match your available tools (project status, list risks, create risks).
Respond conversationally and helpfully, acknowledging their request and suggesting how you can help
with your available capabilities. Keep your response brief and friendly."""

    try:
        response = provider.generate_text(message, system=system_prompt)
        return response
    except Exception:
        return (
            "I couldn't understand that request. I specialize in:\n"
            "• Checking project status • Listing project risks • Creating new risks\n"
            "Try asking about a specific project (e.g., 'What's the status of PRJ-001?')"
        )


def format_response(state: AgentState) -> AgentState:
    if state.next_action == NextAction.AWAIT_APPROVAL:
        return state

    if state.answer:
        return _evolve(state, next_action=NextAction.DONE)

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
    history: list[dict] | None = None,
    provider: "LLMProvider | None" = None,
    on_status: "StatusCallback | None" = None,
) -> AgentState:
    """Drive a request from raw message through to a final or
    approval-pending AgentState."""

    def emit(status: str) -> None:
        if on_status:
            on_status(status)

    emit("🔍 Analyzing your request...")
    state = parse_intent(message, classify, history)
    emit(f"✓ Intent: {state.intent.upper()}")

    state = route_decision(state)

    if state.next_action == NextAction.EXECUTE_READ_TOOL:
        emit(f"🔧 Executing: {state.selected_tool}")
        state = execute_read_tool(state, tool_registry)
    elif state.next_action == NextAction.EXECUTE_WRITE_TOOL:
        emit(f"⚠️ Write action requires approval: {state.selected_tool}")
        state = execute_write_tool(state, tool_registry)
    elif state.next_action == NextAction.GENERATE_RESPONSE:
        emit("💬 Generating response...")
        state = generate_response(state, message, provider)

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
