from functools import lru_cache

from app.approvals.store import ApprovalStore, build_default_store
from app.providers.intent_classifier import get_default_classifier
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
