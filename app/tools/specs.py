"""OpenAI function-calling tool definitions for every tool in app/tools.

An LLM-backed classifier (replacing app.runtime.default_classify) hands these
to the chat completions API's `tools` parameter so the model can pick a tool
and its arguments directly, instead of keyword matching. The classifier is
still responsible for mapping the returned tool name back to an `intent` and
`tool_input` matching app.runtime.IntentClassifier's shape.

Each spec also carries the MCP-style tool-boundary metadata the spec asks
for — permission, side-effect level, timeout, retry limit, audit category —
so app.security reads the same registry the LLM does rather than duplicating
the tool list.
"""

from typing import Literal, TypedDict

from pydantic import BaseModel

from app.tools.schemas import (
    BudgetSummaryInput,
    CreateRiskDraftInput,
    ListProjectTasksInput,
    ListRisksInput,
    ProjectStatusInput,
    SprintProgressInput,
)

SideEffect = Literal["read", "write"]


class ToolSpec(TypedDict):
    type: str
    function: dict


class ToolMeta(BaseModel):
    """Permission-scoping metadata for the tool gateway (app.security)."""

    name: str
    permission: str
    side_effect: SideEffect
    timeout_s: float = 10.0
    retry_limit: int = 2
    audit_category: str


_TOOL_INPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "get_project_status": ProjectStatusInput,
    "list_risks": ListRisksInput,
    "create_risk": CreateRiskDraftInput,
    "list_project_tasks": ListProjectTasksInput,
    "get_sprint_progress": SprintProgressInput,
    "get_budget_summary": BudgetSummaryInput,
}

_TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_project_status": (
        "Check project details: name, stage, owner, and overall status summary. "
        "Use when user asks about project status, progress, or overview."
    ),
    "list_risks": (
        "View all existing risks for a project. "
        "Use when user asks to see, list, view, or check what risks exist for a project."
    ),
    "create_risk": (
        "Add a new risk to a project with title, severity (low/medium/high), and optional description. "
        "Requires approval. Use when user wants to add, create, log, record, or submit a new risk. "
        "Call this even when the user has not said which project, title, or severity they want — "
        "pass only the fields they did supply and leave the rest out. The approval step shows the "
        "user a form to fill in the missing fields, so never withhold the call to ask for them."
    ),
    "list_project_tasks": (
        "List tasks for a project, optionally filtered by status "
        "(open, in_progress, blocked, done, cancelled, overdue) and limited to a count. "
        "Use when the user asks what tasks exist, what's blocked, or what's overdue."
    ),
    "get_sprint_progress": (
        "Get the current delivery iteration's progress for a project: committed, "
        "completed, open, and overdue task counts, and completion percentage. "
        "Use when the user asks about sprint progress, velocity, or iteration status."
    ),
    "get_budget_summary": (
        "Get the financial summary for a project: planned budget, actual cost, "
        "committed cost, remaining budget, and variance. Requires finance read "
        "permission. Use when the user asks about budget, cost, spend, or variance."
    ),
}

# MCP-style boundary metadata, one entry per tool. `permission` values are the
# vocabulary app.security.permissions checks against.
TOOL_META: dict[str, ToolMeta] = {
    "get_project_status": ToolMeta(
        name="get_project_status", permission="project.read", side_effect="read",
        audit_category="project_read",
    ),
    "list_risks": ToolMeta(
        name="list_risks", permission="project.risk.read", side_effect="read",
        audit_category="risk_read",
    ),
    "create_risk": ToolMeta(
        name="create_risk", permission="project.risk.create", side_effect="write",
        retry_limit=0,  # a write is never blindly retried; see app/graph/retry.py
        audit_category="risk_write",
    ),
    "list_project_tasks": ToolMeta(
        name="list_project_tasks", permission="project.read", side_effect="read",
        audit_category="task_read",
    ),
    "get_sprint_progress": ToolMeta(
        name="get_sprint_progress", permission="project.read", side_effect="read",
        audit_category="sprint_read",
    ),
    "get_budget_summary": ToolMeta(
        name="get_budget_summary", permission="project.finance.read", side_effect="read",
        audit_category="budget_read",
    ),
}


def _tool_spec(name: str) -> ToolSpec:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": _TOOL_DESCRIPTIONS[name],
            "parameters": _TOOL_INPUT_SCHEMAS[name].model_json_schema(),
        },
    }


def get_openai_tool_specs() -> list[ToolSpec]:
    return [_tool_spec(name) for name in _TOOL_INPUT_SCHEMAS]


def get_tool_meta(name: str) -> ToolMeta | None:
    return TOOL_META.get(name)


def permission_for(tool_name: str) -> str | None:
    meta = TOOL_META.get(tool_name)
    return meta.permission if meta else None
