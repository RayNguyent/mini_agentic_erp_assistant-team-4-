"""Auth enforcement, readiness/tools introspection, role-scoped permission
denial, and project-document RAG reachable end-to-end through /chat."""

import pytest

from app.api.dependencies import get_retriever
from app.api.main import app
from app.rag.documents import Chunk
from app.rag.retrieve import Retriever
from app.rag.store import ChunkStore
from app.rag.vector_index import NullVectorIndex
from tests.conftest import AUDIT_TOKEN, DEV_TOKEN, PM_TOKEN


# --- auth ---------------------------------------------------------------


def test_chat_without_a_token_is_unauthorized(client):
    response = client.post("/chat", json={"message": "hi"}, headers={"Authorization": ""})
    assert response.status_code == 401


def test_chat_with_an_unrecognised_token_is_unauthorized(client):
    response = client.post(
        "/chat", json={"message": "hi"}, headers={"Authorization": "Bearer not-a-real-token"}
    )
    assert response.status_code == 401


def test_health_and_readiness_do_not_require_auth(client):
    # Operational endpoints must be reachable without a session so a load
    # balancer / uptime check can hit them.
    assert client.get("/health", headers={"Authorization": ""}).status_code == 200
    assert client.get("/readiness", headers={"Authorization": ""}).status_code == 200


# --- readiness ------------------------------------------------------------


def test_readiness_reports_deterministic_mode_and_index_status(client):
    body = client.get("/readiness").json()
    assert body["llm_provider"] == "deterministic"
    assert body["credential_configured"] is False
    assert body["status"] in ("ok", "degraded")
    assert "rag_index_chunks" in body


# --- tools ------------------------------------------------------------------


def test_tools_endpoint_lists_all_six_with_boundary_metadata(client):
    tools = {t["name"]: t for t in client.get("/tools").json()["tools"]}
    assert set(tools) == {
        "get_project_status", "list_risks", "create_risk",
        "list_project_tasks", "get_sprint_progress", "get_budget_summary",
    }
    assert tools["create_risk"]["side_effect"] == "write"
    assert tools["get_budget_summary"]["permission"] == "project.finance.read"


# --- role-scoped permission denial -------------------------------------------


def test_developer_is_denied_budget_summary(client):
    response = client.post(
        "/chat",
        json={"message": "What is the budget for PRJ-001?"},
        headers={"Authorization": f"Bearer {DEV_TOKEN}"},
    )
    body = response.json()
    assert body["error_code"] == "FORBIDDEN"


def test_project_manager_is_allowed_budget_summary(client):
    response = client.post(
        "/chat",
        json={"message": "What is the budget for PRJ-001?"},
        headers={"Authorization": f"Bearer {PM_TOKEN}"},
    )
    body = response.json()
    assert body["error_code"] is None
    assert body["tool_used"] == "get_budget_summary"


def test_developer_is_denied_creating_a_risk(client):
    response = client.post(
        "/chat",
        json={"message": "Create a risk for PRJ-001"},
        headers={"Authorization": f"Bearer {DEV_TOKEN}"},
    )
    body = response.json()
    assert body["error_code"] == "FORBIDDEN"
    assert body["approval_required"] is False  # never reached the gate


# --- RAG reachable end-to-end through /chat ----------------------------------


@pytest.fixture
def rag_client(client):
    """Overrides the (empty, by default) retriever with one seeded chunk, so
    a document question routed through /chat has something to find."""
    chunks = [
        Chunk(
            chunk_id="POL#c00", doc_id="POL", title="Risk Policy",
            text="Risks are rated low, medium, or high severity. Only the risk owner or the project manager may close a risk.",
            classification="public",
        ),
        Chunk(
            chunk_id="FIN#c00", doc_id="FIN", title="Budget Report",
            text="The restricted licensing negotiation target is a 14 percent reduction.",
            classification="restricted",
        ),
    ]
    retriever = Retriever(ChunkStore(chunks=chunks), vector_index=NullVectorIndex())
    app.dependency_overrides[get_retriever] = lambda: retriever
    yield client
    del app.dependency_overrides[get_retriever]


def test_a_document_question_reaches_rag_through_chat_with_citations(rag_client):
    response = rag_client.post(
        "/chat", json={"message": "What are the risk severity ratings and who can close a risk?"}
    )
    body = response.json()
    assert body["citations"]
    assert body["citations"][0]["chunk_id"] == "POL#c00"
    assert body["route"] in ("multi_agent",)


def test_a_restricted_document_is_never_cited_for_a_developer(rag_client):
    response = rag_client.post(
        "/chat",
        json={"message": "What is the licensing negotiation target?"},
        headers={"Authorization": f"Bearer {DEV_TOKEN}"},
    )
    body = response.json()
    assert body["citations"] == []


def test_the_same_restricted_question_is_answered_for_an_auditor(rag_client):
    response = rag_client.post(
        "/chat",
        json={"message": "What is the licensing negotiation target?"},
        headers={"Authorization": f"Bearer {AUDIT_TOKEN}"},
    )
    body = response.json()
    assert any(c["chunk_id"] == "FIN#c00" for c in body["citations"])


def test_an_unanswerable_document_question_refuses_without_inventing_a_citation(rag_client):
    response = rag_client.post("/chat", json={"message": "What is the weather in Paris?"})
    body = response.json()
    assert body["citations"] == []
    assert body["error_code"] is None  # a refusal is a valid answer, not an error
