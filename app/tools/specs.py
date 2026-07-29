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
    "get_project_status": "Look up the current status of a single ERP project by its project code.",
    "list_risks": "List every risk recorded against a single ERP project.",
    "create_risk": (
        "Log a new risk against an ERP project. This is a write action that "
        "requires human approval before it takes effect."
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
