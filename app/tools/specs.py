"""OpenAI function-calling tool definitions for every tool in app/tools.

An LLM-backed classifier (replacing app.runtime.default_classify) hands these
to the chat completions API's `tools` parameter so the model can pick a tool
and its arguments directly, instead of keyword matching. The classifier is
still responsible for mapping the returned tool name back to an `intent` and
`tool_input` matching app.runtime.IntentClassifier's shape.
"""

from typing import TypedDict

from pydantic import BaseModel

from app.tools.schemas import CreateRiskInput, ListRisksInput, ProjectStatusInput


class ToolSpec(TypedDict):
    type: str
    function: dict


_TOOL_INPUT_SCHEMAS: dict[str, type[BaseModel]] = {
    "get_project_status": ProjectStatusInput,
    "list_risks": ListRisksInput,
    "create_risk": CreateRiskInput,
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
        "Requires approval. Use when user wants to add, create, log, record, or submit a new risk."
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
