import logging

from pydantic import BaseModel, ValidationError

from app.errors import ToolError
from app.providers.base import LLMProvider
from app.providers.config import ProviderSettings
from app.providers.openai_provider import OpenAIProvider
from app.runtime import READ_TOOLS, WRITE_TOOLS, IntentClassifier, default_classify
from app.state import IntentType
from app.tools.specs import get_openai_tool_specs

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an intent classifier for a project-management assistant.
Classify the user's message into exactly one of these intents:
- project_status: asking about a project's overall status
- list_risks: asking to see/list risks for a project
- create_risk: asking to create/add/log a new risk for a project
- unsupported: anything else (refuse rather than guess)

If a project code matching the pattern LETTERS-NUMBERS (e.g. PRJ-001) appears in the
message, extract it; otherwise use null.

Respond with ONLY a JSON object of this exact shape, no other text:
{"intent": "<one of the four values above>", "project_code": "<code or null>"}
"""


_TOOL_CALL_SYSTEM_PROMPT = """You are an intelligent routing assistant for an ERP project management system.

Your job is to understand what the user wants and call exactly one tool that matches their intent:

TOOLS AVAILABLE:
- get_project_status: When user asks to CHECK, VIEW, LOOK UP, or GET STATUS of a project
- list_risks: When user asks to VIEW, LIST, SHOW, or SEE risks for a project
- create_risk: When user wants to ADD, CREATE, LOG, RECORD, or SUBMIT a NEW risk to a project

IMPORTANT:
- If the user wants to MODIFY/WRITE/CREATE something, use create_risk
- If the user just wants to VIEW/READ/LIST information, use get_project_status or list_risks
- If none apply, do NOT call any tool
- Extract project codes from context (e.g., "project 2" → "PRJ-002", "PRJ-001" stays "PRJ-001")
- Always pad project numbers to 3 digits (2 → 002, 001 → 001)
- Use prior conversation to resolve follow-ups: "How about PRJ-003?" repeats the previous tool with new project
"""

_TOOL_NAME_TO_INTENT = {**{v: k for k, v in READ_TOOLS.items()}, **{v: k for k, v in WRITE_TOOLS.items()}}


class IntentClassification(BaseModel):
    intent: IntentType
    project_code: str | None = None


def make_llm_classifier(
    provider: LLMProvider, fallback: IntentClassifier = default_classify
) -> IntentClassifier:
    """Wraps an LLMProvider as an IntentClassifier.

    Any provider failure or malformed/invalid response falls back to the
    deterministic classifier immediately (no retry loop here — the fallback
    *is* the safety net) so a model outage never breaks routing.
    """

    def classify(message: str, history: list[dict] | None = None) -> tuple[str, dict]:
        try:
            raw = provider.generate(message, system=_SYSTEM_PROMPT, history=history)
            parsed = IntentClassification.model_validate_json(raw)
        except (ToolError, ValidationError):
            return fallback(message, history)

        tool_input = {"project_code": parsed.project_code} if parsed.project_code else {}
        return parsed.intent.value, tool_input

    return classify


def make_tool_calling_classifier(provider: LLMProvider) -> IntentClassifier:
    """Wraps an LLMProvider's tool-calling ability as an IntentClassifier.

    Hands the model the real tool schemas and trusts its decision-making.
    Only falls back to unsupported if the model declines or errors.
    """
    tools = get_openai_tool_specs()

    def classify(message: str, history: list[dict] | None = None) -> tuple[str, dict]:
        try:
            call = provider.generate_tool_call(
                message, tools=tools, system=_TOOL_CALL_SYSTEM_PROMPT, history=history
            )
        except ToolError as exc:
            logger.warning("LLM provider error: %s (returning unsupported)", exc)
            return "unsupported", {}

        if call is None:
            logger.info("LLM declined to call any tool")
            return "unsupported", {}

        intent = _TOOL_NAME_TO_INTENT.get(call.name)
        if intent is None:
            logger.warning("LLM called unknown tool: %s (returning unsupported)", call.name)
            return "unsupported", {}

        logger.info("LLM classified: intent=%s project_code=%s", intent, call.arguments.get("project_code"))
        return intent, call.arguments

    return classify


def get_default_classifier(settings: ProviderSettings | None = None) -> IntentClassifier:
    """Reads provider config and returns whichever classifier is configured."""
    settings = settings or ProviderSettings.from_env()
    if settings.llm_provider == "openai" and settings.openai_api_key:
        provider = OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout_s=settings.openai_timeout_s,
        )
        return make_tool_calling_classifier(provider)
    return default_classify
