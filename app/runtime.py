"""The single-agent graph: intent -> route -> tool/conversation -> answer.

Node functions stay independently callable (`route_decision(state)`) so they
can be unit-tested without standing up a graph; `build_graph()` wires the same
functions into the engine that `run()`/`resume()` drive. The multi-agent
supervisor in app/agents composes these same nodes into a wider graph.
"""

import re
import time
from typing import TYPE_CHECKING, Protocol

from app.errors import ErrorCode, NonRetryableToolError, RetryableToolError
from app.graph.engine import Graph, GraphContext, evolve
from app.graph.retry import RetryPolicy, retry_call
from app.state import MAX_RETRIES, AgentState, NextAction, Route

if TYPE_CHECKING:
    from app.providers.base import LLMProvider

# intent -> tool name. Extend as new tools land in app/tools.
READ_TOOLS = {
    "project_status": "get_project_status",
    "list_risks": "list_risks",
    "list_project_tasks": "list_project_tasks",
    "sprint_progress": "get_sprint_progress",
    "budget_summary": "get_budget_summary",
}
WRITE_TOOLS = {
    "create_risk": "create_risk",
}

DEFAULT_RETRY_POLICY = RetryPolicy(max_retries=MAX_RETRIES)

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


# Backwards-compatible alias: the engine's `evolve` used to live here.
_evolve = evolve


def default_classify(message: str, history: list[dict] | None = None) -> tuple[str, dict]:
    """Deterministic, offline keyword classifier.

    The fallback whenever no LLM provider is configured or a provider-backed
    classifier misbehaves. `history` is accepted (to match IntentClassifier)
    but unused — this classifier has no way to reason over prior turns.
    """
    lowered = message.lower()
    match = _PROJECT_CODE_RE.search(message.upper())
    project_code = _normalize_project_code(match.group(0)) if match else None

    if "risk" in lowered and any(w in lowered for w in ("create", "add", "log", "new")):
        intent = "create_risk"
    elif "risk" in lowered:
        intent = "list_risks"
    elif "budget" in lowered or "cost" in lowered or "spend" in lowered or "variance" in lowered:
        intent = "budget_summary"
    elif "sprint" in lowered or "velocity" in lowered or "iteration" in lowered:
        intent = "sprint_progress"
    elif "task" in lowered:
        intent = "list_project_tasks"
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


# --- Nodes ------------------------------------------------------------------


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
        return evolve(
            state,
            selected_tool=READ_TOOLS[state.intent],
            route=Route.TOOL,
            next_action=NextAction.EXECUTE_READ_TOOL,
            status="Executing tool...",
        )
    if state.intent in WRITE_TOOLS:
        return evolve(
            state,
            selected_tool=WRITE_TOOLS[state.intent],
            approval_required=True,
            route=Route.TOOL,
            risk_level="high",
            next_action=NextAction.EXECUTE_WRITE_TOOL,
            status="Requesting approval...",
        )
    return evolve(
        state,
        route=Route.CONVERSATION,
        next_action=NextAction.GENERATE_RESPONSE,
        status="Generating helpful response...",
    )


def _run_tool_with_retry(
    state: AgentState,
    tool_registry: ToolRegistry,
    *,
    policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    sleep=time.sleep,
    ctx: GraphContext | None = None,
) -> AgentState:
    tool_fn = tool_registry.get(state.selected_tool)
    if tool_fn is None:
        return evolve(
            state,
            next_action=NextAction.FORMAT_RESPONSE,
            error_code=ErrorCode.NOT_FOUND,
            error_message=f"No tool registered for '{state.selected_tool}'.",
        )

    tool_input = state.tool_input or {}
    if isinstance(tool_input.get("project_code"), str):
        tool_input = {**tool_input, "project_code": _normalize_project_code(tool_input["project_code"])}

    retries = 0

    def _on_retry(attempt: int, exc: Exception) -> None:
        nonlocal retries
        retries = attempt
        if ctx is not None:
            ctx.emit(f"⟳ Retry {attempt}/{policy.max_retries} after {type(exc).__name__}")

    try:
        output, _ = retry_call(
            lambda: tool_fn(tool_input),
            policy=policy,
            # Only transient failures are retried; a NotFoundError or a
            # validation failure is deterministic and would fail identically.
            is_retryable=lambda exc: isinstance(exc, RetryableToolError),
            on_retry=_on_retry,
            sleep=sleep,
        )
    except (RetryableToolError, NonRetryableToolError) as exc:
        return evolve(
            state,
            next_action=NextAction.FORMAT_RESPONSE,
            error_code=exc.code,
            error_message=exc.message,
            retry_count=retries,
        )

    return evolve(
        state,
        tool_output=output,
        next_action=NextAction.FORMAT_RESPONSE,
        error_code=None,
        error_message=None,
        retry_count=retries,
    )


def execute_read_tool(state: AgentState, tool_registry: ToolRegistry, **kwargs) -> AgentState:
    return _run_tool_with_retry(state, tool_registry, **kwargs)


def execute_write_tool(state: AgentState, tool_registry: ToolRegistry, **kwargs) -> AgentState:
    if state.approved is None:
        return evolve(state, next_action=NextAction.AWAIT_APPROVAL)
    if state.approved is False:
        return evolve(
            state,
            next_action=NextAction.FORMAT_RESPONSE,
            error_code=ErrorCode.APPROVAL_REJECTED,
            error_message="The requested action was rejected during approval.",
        )
    return _run_tool_with_retry(state, tool_registry, **kwargs)


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


def _render_task(task: dict) -> str:
    overdue = " ⚠ overdue" if task.get("overdue") else ""
    return (
        f"• [{task.get('task_id', '?')}] {task.get('title', '?')} "
        f"({task.get('state', '?')}) — due {task.get('deadline') or 'unscheduled'}{overdue}"
    )


def _render_list_project_tasks(output: dict) -> str:
    tasks = output.get("tasks", [])
    if not tasks:
        return f"No matching tasks for {output.get('project_code', '?')}."
    header = f"Tasks for {output.get('project_code', '?')} ({len(tasks)} shown):"
    return "\n".join([header, *(_render_task(task) for task in tasks)])


def _render_sprint_progress(output: dict) -> str:
    return (
        f"Sprint {output.get('iteration_label', '?')} for {output.get('project_code', '?')} "
        f"({output.get('start_date', '?')} → {output.get('end_date', '?')})\n"
        f"• Mapping profile: {output.get('mapping_profile', '?')}\n"
        f"• Committed: {output.get('committed', 0)} | "
        f"Completed: {output.get('completed', 0)} | "
        f"Open: {output.get('open', 0)} | "
        f"Overdue: {output.get('overdue', 0)}\n"
        f"• Completion: {output.get('completion_pct', 0)}%"
    )


def _render_budget_summary(output: dict) -> str:
    currency = output.get("currency", "")
    flags = output.get("completeness_flags", [])
    lines = [
        f"Budget for {output.get('project_code', '?')} ({currency}, as of {output.get('as_of', '?')})",
        f"• Planned: {output.get('planned_budget')}",
        f"• Actual cost: {output.get('actual_cost')}",
        f"• Committed: {output.get('committed_cost')}",
        f"• Forecast revenue: {output.get('forecast_revenue')}",
        f"• Remaining: {output.get('remaining')} | Variance: {output.get('variance')}",
    ]
    if flags:
        # Surfaced rather than silently zero-filled: an incomplete budget that
        # looks complete is the failure mode this tool exists to avoid.
        lines.append(f"• Incomplete data: {', '.join(flags)}")
    return "\n".join(lines)


_TOOL_RENDERERS = {
    "get_project_status": _render_project_status,
    "list_risks": _render_list_risks,
    "create_risk": _render_create_risk,
    "list_project_tasks": _render_list_project_tasks,
    "get_sprint_progress": _render_sprint_progress,
    "get_budget_summary": _render_budget_summary,
}


def _render_tool_output(tool_name: str | None, output: dict) -> str:
    renderer = _TOOL_RENDERERS.get(tool_name or "")
    if renderer is None:
        return f"{tool_name}: {output}"
    return renderer(output)


def generate_response(
    state: AgentState, message: str, provider: "LLMProvider | None" = None
) -> AgentState:
    """Generate a conversational response when no tool applies."""
    response = generate_conversational_response(message, provider)
    return evolve(
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
        return evolve(state, next_action=NextAction.DONE)

    if state.error_code is not None:
        answer = state.error_message or "The request could not be completed."
    elif state.tool_output is not None:
        answer = _render_tool_output(state.selected_tool, state.tool_output)
    else:
        answer = "No result available."

    return evolve(state, answer=answer, next_action=NextAction.DONE)


# --- Graph wiring -----------------------------------------------------------


def _parse_intent_node(state: AgentState, ctx: GraphContext) -> AgentState:
    ctx.emit("🔍 Analyzing your request...")
    parsed = parse_intent(
        ctx.require("message"),
        ctx.get("classify", default_classify),
        ctx.get("history"),
    )
    ctx.emit(f"✓ Intent: {parsed.intent.upper()}")
    # Carry request-scoped identity fields the classifier does not know about.
    return evolve(
        parsed,
        request_id=state.request_id,
        actor=state.actor,
        idempotency_key=state.idempotency_key,
    )


def _route_decision_node(state: AgentState, ctx: GraphContext) -> AgentState:
    routed = route_decision(state)
    if routed.next_action == NextAction.EXECUTE_READ_TOOL:
        ctx.emit(f"🔧 Executing: {routed.selected_tool}")
    elif routed.next_action == NextAction.EXECUTE_WRITE_TOOL:
        ctx.emit(f"⚠️ Write action requires approval: {routed.selected_tool}")
    else:
        ctx.emit("💬 Generating response...")
    return routed


def _execute_read_tool_node(state: AgentState, ctx: GraphContext) -> AgentState:
    return execute_read_tool(
        state,
        ctx.require("tool_registry"),
        policy=ctx.get("retry_policy", DEFAULT_RETRY_POLICY),
        ctx=ctx,
    )


def _execute_write_tool_node(state: AgentState, ctx: GraphContext) -> AgentState:
    return execute_write_tool(
        state,
        ctx.require("tool_registry"),
        policy=ctx.get("retry_policy", DEFAULT_RETRY_POLICY),
        ctx=ctx,
    )


def _generate_response_node(state: AgentState, ctx: GraphContext) -> AgentState:
    return generate_response(state, ctx.require("message"), ctx.get("provider"))


def _format_response_node(state: AgentState, ctx: GraphContext) -> AgentState:
    formatted = format_response(state)
    ctx.emit("✓ Done")
    return formatted


def build_graph() -> Graph:
    """The single-agent graph. AWAIT_APPROVAL is terminal *and* re-enterable:
    `resume()` restarts the driver from that same node once approval is set."""
    return (
        Graph("single_agent")
        .add_node(NextAction.PARSE_INTENT, _parse_intent_node)
        .add_node(NextAction.ROUTE_DECISION, _route_decision_node)
        .add_node(NextAction.EXECUTE_READ_TOOL, _execute_read_tool_node)
        .add_node(NextAction.EXECUTE_WRITE_TOOL, _execute_write_tool_node)
        .add_node(NextAction.GENERATE_RESPONSE, _generate_response_node)
        .add_node(NextAction.FORMAT_RESPONSE, _format_response_node)
        .add_terminal(NextAction.DONE, NextAction.AWAIT_APPROVAL)
    )


GRAPH = build_graph()


def run(
    message: str,
    tool_registry: ToolRegistry,
    classify: IntentClassifier = default_classify,
    history: list[dict] | None = None,
    provider: "LLMProvider | None" = None,
    on_status: "StatusCallback | None" = None,
    *,
    actor=None,
    request_id: str | None = None,
    trace=None,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
) -> AgentState:
    """Drive a request from raw message through to a final or
    approval-pending AgentState."""
    ctx = GraphContext(
        trace=trace,
        message=message,
        tool_registry=tool_registry,
        classify=classify,
        history=history,
        provider=provider,
        on_status=on_status,
        retry_policy=retry_policy,
    )
    initial = AgentState(
        intent="unknown",
        next_action=NextAction.PARSE_INTENT,
        request_id=request_id,
        actor=actor,
    )
    return GRAPH.invoke(initial, ctx)


def resume(
    state: AgentState,
    tool_registry: ToolRegistry,
    *,
    on_status: "StatusCallback | None" = None,
    trace=None,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
) -> AgentState:
    """Continue a write-tool flow after /approve or /reject has set
    `state.approved`. Re-enters the same graph at EXECUTE_WRITE_TOOL."""
    if state.next_action != NextAction.AWAIT_APPROVAL:
        raise ValueError(
            f"resume() called on a state that is not awaiting approval: {state.next_action}"
        )
    ctx = GraphContext(
        trace=trace,
        message="",
        tool_registry=tool_registry,
        on_status=on_status,
        retry_policy=retry_policy,
    )
    return GRAPH.invoke(
        evolve(state, next_action=NextAction.EXECUTE_WRITE_TOOL),
        ctx,
    )
