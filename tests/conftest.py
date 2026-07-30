import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import (
    get_approval_store,
    get_intent_classifier,
    get_tool_registry,
)
from app.api.main import app
from app.approvals.store import build_default_store
from app.runtime import default_classify
from app.tools.registry import build_default_registry


@pytest.fixture
def client():
    """TestClient with a fresh registry and approval store per test.

    The app's own dependencies are process-lifetime singletons, so without
    these overrides a risk created in one test would leak into the next.

    The classifier is pinned to the deterministic keyword one: with the real
    dependency these tests reach a live LLM, which makes them slow, network-
    dependent, and flaky on the model's wording of the day.
    """
    registry = build_default_registry()
    store = build_default_store()
    app.dependency_overrides[get_tool_registry] = lambda: registry
    app.dependency_overrides[get_approval_store] = lambda: store
    app.dependency_overrides[get_intent_classifier] = lambda: default_classify

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
