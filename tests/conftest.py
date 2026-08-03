import pytest
from fastapi.testclient import TestClient

from app.agents.graph import default_agent_registry
from app.api.dependencies import (
    get_agent_registry,
    get_approval_store,
    get_audit_log,
    get_authenticator,
    get_intent_classifier,
    get_memory_store,
    get_retriever,
    get_tool_registry,
    get_trace_store,
)
from app.api.main import app
from app.approvals.store import build_default_store
from app.memory.store import MemoryStore
from app.observability.trace import TraceStore
from app.rag.retrieve import Retriever
from app.rag.store import ChunkStore
from app.rag.vector_index import NullVectorIndex
from app.runtime import default_classify
from app.security.audit import AuditLog
from app.security.auth import TokenAuthenticator
from app.tools.registry import build_default_registry

PM_TOKEN = "test-pm-token"
DEV_TOKEN = "test-dev-token"
AUDIT_TOKEN = "test-audit-token"


@pytest.fixture
def client():
    """TestClient with a fresh registry, approval store, and empty RAG index
    per test, authenticated by default as a project_manager.

    project_manager was picked as the default because it is a superset of the
    permissions every non-auth-focused test needs (read + create_risk); tests
    that specifically exercise a role boundary override the header directly.

    The app's own dependencies are process-lifetime singletons, so without
    these overrides a risk created in one test would leak into the next.

    The classifier is pinned to the deterministic keyword one: with the real
    dependency these tests reach a live LLM, which makes them slow, network-
    dependent, and flaky on the model's wording of the day.
    """
    registry = build_default_registry()
    store = build_default_store()
    retriever = Retriever(ChunkStore(), vector_index=NullVectorIndex())  # empty: no doc fixtures per-test
    authenticator = TokenAuthenticator({
        PM_TOKEN: ("test.pm", "project_manager"),
        DEV_TOKEN: ("test.dev", "developer"),
        AUDIT_TOKEN: ("test.audit", "auditor"),
    })
    audit = AuditLog.__new__(AuditLog)
    import threading

    audit._lock = threading.Lock()
    audit._rows = []
    from pathlib import Path

    audit._path = Path("data/.test-audit.jsonl")  # overwritten below to avoid disk noise
    audit._path.parent.mkdir(parents=True, exist_ok=True)

    trace_store = TraceStore(path=Path("data/.test-traces.jsonl"))
    memory_store = MemoryStore(path=Path("data/.test-memory.jsonl"))

    app.dependency_overrides[get_tool_registry] = lambda: registry
    app.dependency_overrides[get_approval_store] = lambda: store
    app.dependency_overrides[get_intent_classifier] = lambda: default_classify
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_agent_registry] = lambda: default_agent_registry()
    app.dependency_overrides[get_authenticator] = lambda: authenticator
    app.dependency_overrides[get_audit_log] = lambda: audit
    app.dependency_overrides[get_trace_store] = lambda: trace_store
    app.dependency_overrides[get_memory_store] = lambda: memory_store

    with TestClient(app, headers={"Authorization": f"Bearer {PM_TOKEN}"}) as test_client:
        yield test_client

    app.dependency_overrides.clear()
