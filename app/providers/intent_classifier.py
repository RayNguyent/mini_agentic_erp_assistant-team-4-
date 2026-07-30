from pydantic import BaseModel, ValidationError

from app.errors import ToolError
from app.providers.base import LLMProvider
from app.providers.config import ProviderSettings
from app.providers.openai_provider import OpenAIProvider
from app.runtime import IntentClassifier, default_classify
from app.state import IntentType

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

    def classify(message: str) -> tuple[str, dict]:
        try:
            raw = provider.generate(message, system=_SYSTEM_PROMPT)
            parsed = IntentClassification.model_validate_json(raw)
        except (ToolError, ValidationError):
            return fallback(message)

        tool_input = {"project_code": parsed.project_code} if parsed.project_code else {}
        return parsed.intent.value, tool_input

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
        return make_llm_classifier(provider)
    return default_classify
