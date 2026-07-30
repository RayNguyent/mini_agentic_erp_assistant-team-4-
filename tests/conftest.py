import pytest
from fastapi.testclient import TestClient

from app.api.dependencies import get_approval_store, get_tool_registry
from app.api.main import app
from app.approvals.store import build_default_store
from app.tools.registry import build_default_registry


@pytest.fixture
def client():
    """TestClient with a fresh registry and approval store per test.

    The app's own dependencies are process-lifetime singletons, so without
    these overrides a risk created in one test would leak into the next.
    """
    registry = build_default_registry()
    store = build_default_store()
    app.dependency_overrides[get_tool_registry] = lambda: registry
    app.dependency_overrides[get_approval_store] = lambda: store

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
