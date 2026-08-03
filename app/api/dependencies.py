from functools import lru_cache

from fastapi import Depends, Header

from app.agents.graph import default_agent_registry
from app.approvals.store import ApprovalStore, build_default_store
from app.providers.base import LLMProvider
from app.providers.config import ProviderSettings
from app.providers.intent_classifier import get_default_classifier
from app.providers.ollama_provider import OllamaProvider
from app.providers.openai_provider import OpenAIProvider
from app.rag.retrieve import Retriever, build_retriever
from app.runtime import IntentClassifier, ToolRegistry
from app.security.audit import AuditLog, build_default_audit_log
from app.security.auth import AuthError, TokenAuthenticator, build_default_authenticator
from app.security.permissions import check as check_permission
from app.state import Actor
from app.tools.registry import build_default_registry

# These must stay process-lifetime singletons, not per-request. The approval
# store holds pending states between the /chat that creates one and the
# /approve that resolves it, and MockERPProvider keeps created risks in memory
# only — rebuilding either per request would silently lose both. The memory
# store needs the same treatment: long-term memory has no persistence besides
# this process's in-memory index (plus its JSONL log), and working memory is
# explicitly session/process-scoped by design (app.memory.working).


@lru_cache(maxsize=1)
def get_tool_registry() -> ToolRegistry:
    return build_default_registry()


@lru_cache(maxsize=1)
def _shared_provider_settings() -> ProviderSettings:
    return ProviderSettings.from_env()


@lru_cache(maxsize=1)
def get_llm_provider() -> LLMProvider | None:
    """The single shared LLM adapter — generation, classification, planning,
    and RAG grounding all use this instance rather than each constructing
    their own client (the earlier asymmetry between get_default_classifier
    and get_llm_provider building separate OpenAIProvider objects)."""
    settings = _shared_provider_settings()
    if not settings.is_live:
        return None
    if settings.llm_provider == "ollama":
        return OllamaProvider(
            model=settings.openai_model, base_url=settings.resolved_base_url,
            timeout_s=settings.openai_timeout_s,
        )
    return OpenAIProvider(
        api_key=settings.openai_api_key or "not-needed",
        model=settings.openai_model,
        base_url=settings.resolved_base_url,
        timeout_s=settings.openai_timeout_s,
    )


@lru_cache(maxsize=1)
def get_intent_classifier() -> IntentClassifier:
    return get_default_classifier(_shared_provider_settings())


@lru_cache(maxsize=1)
def get_approval_store() -> ApprovalStore:
    return build_default_store()


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    return build_retriever()


@lru_cache(maxsize=1)
def get_agent_registry() -> dict:
    return default_agent_registry()


@lru_cache(maxsize=1)
def get_authenticator() -> TokenAuthenticator:
    return build_default_authenticator()


@lru_cache(maxsize=1)
def get_audit_log() -> AuditLog:
    return build_default_audit_log()


@lru_cache(maxsize=1)
def get_trace_store():
    from app.observability.trace import build_default_trace_store

    return build_default_trace_store()


@lru_cache(maxsize=1)
def get_memory_store():
    from app.memory.store import build_default_memory_store

    return build_default_memory_store()


def get_current_actor(
    authorization: str | None = Header(default=None),
    authenticator: TokenAuthenticator = Depends(get_authenticator),
) -> Actor:
    """Every request carries an identity — final-project.pdf §7:
    "Each request must carry a user identity and role." A missing or bad
    bearer token raises AuthError, mapped to HTTP 401 in app.api.main.
    """
    token = None
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:].strip()
    return authenticator.authenticate(token)


def get_authorize_fn():
    """The `authorize(actor, permission) -> bool` callable
    app.agents.base.authorize (via GraphContext) checks against."""
    return check_permission
