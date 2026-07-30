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


_TOOL_CALL_SYSTEM_PROMPT = """You are a routing assistant for a project-management ERP agent.
Call exactly one of the available tools that best matches the user's request, filling in
every required argument from the conversation. If none of the tools apply, do not call
any tool. Use prior turns in the conversation to resolve follow-ups: if the user's message
is just a bare project code or a short phrase like "how about X" / "what about X" with no
other detail, treat it as a repeat of the SAME tool call as the immediately preceding turn,
just with the new project code substituted in.
When filling project_code, use the canonical LETTERS-DIGITS form (e.g. PRJ-001) even if
the user wrote it differently (e.g. "prj 001" or "PRJ_001").
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


def make_tool_calling_classifier(
    provider: LLMProvider, fallback: IntentClassifier = default_classify
) -> IntentClassifier:
    """Wraps an LLMProvider's tool-calling ability as an IntentClassifier.

    Unlike make_llm_classifier (which only ever extracts intent +
    project_code via a JSON-shaped prompt), this hands the model the real
    tool schemas from app.tools.specs and lets it fill in every argument
    directly — e.g. create_risk's risk_payload (title/severity/description)
    — so the resulting tool_input is ready to execute without a second
    round-trip to collect missing fields.
    """
    tools = get_openai_tool_specs()

    def classify(message: str, history: list[dict] | None = None) -> tuple[str, dict]:
        try:
            call = provider.generate_tool_call(
                message, tools=tools, system=_TOOL_CALL_SYSTEM_PROMPT, history=history
            )
        except ToolError as exc:
            logger.info("intent classifier fallback: reason=provider_error error=%s", exc)
            return fallback(message, history)

        if call is None:
            logger.info("intent classified: intent=unsupported tool_input={} (model declined to call a tool)")
            return "unsupported", {}

        intent = _TOOL_NAME_TO_INTENT.get(call.name)
        if intent is None:
            logger.info("intent classifier fallback: reason=unknown_tool_name tool_name=%s", call.name)
            return fallback(message, history)

        logger.info("intent classified: intent=%s tool_input=%s", intent, call.arguments)
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
