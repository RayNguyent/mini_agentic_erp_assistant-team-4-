from functools import lru_cache

from app.approvals.store import ApprovalStore, build_default_store
from app.providers.base import LLMProvider
from app.providers.config import ProviderSettings
from app.providers.intent_classifier import get_default_classifier
from app.providers.openai_provider import OpenAIProvider
from app.runtime import IntentClassifier, ToolRegistry
from app.tools.registry import build_default_registry

# These must stay process-lifetime singletons, not per-request. The approval
# store holds pending states between the /chat that creates one and the
# /approve that resolves it, and MockERPProvider keeps created risks in memory
# only — rebuilding either per request would silently lose both.


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    return build_default_registry()


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    return get_default_classifier()


@lru_cache(maxsize=1)
def get_approval_store() -> ApprovalStore:
    return build_default_store()


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider | None:
    """Get the LLM provider for conversational fallback responses.

    Returns None if not configured, in which case default text responses are used."""
    settings = ProviderSettings.from_env()
    if settings.llm_provider == "openai" and settings.openai_api_key:
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout_s=settings.openai_timeout_s,
        )
    return None
