RISK_PAYLOAD = {"title": "Scope creep", "severity": "medium", "description": "requirements growing"}


def _pending_create_risk(client, project_code="PRJ-001") -> str:
    response = client.post("/chat", json={"message": f"Create a risk for {project_code}"})
    return response.json()["approval_id"]


def test_approve_executes_the_write_tool(client):
    approval_id = _pending_create_risk(client)

    response = client.post(
        "/approve",
        json={"approval_id": approval_id, "tool_input": {"risk_payload": RISK_PAYLOAD}},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["error_code"] is None
    assert body["tool_used"] == "create_risk"
    assert "Scope creep" in body["answer"]


def test_approved_risk_is_visible_to_a_later_list_risks_call(client):
    before = client.post("/chat", json={"message": "Show all risks for PRJ-002"})
    assert "RISK-" not in before.json()["answer"]

    approval_id = _pending_create_risk(client, "PRJ-002")
    client.post(
        "/approve",
        json={"approval_id": approval_id, "tool_input": {"risk_payload": RISK_PAYLOAD}},
    )

    after = client.post("/chat", json={"message": "Show all risks for PRJ-002"})
    assert "Scope creep" in after.json()["answer"]


def test_reject_does_not_execute_the_write_tool(client):
    approval_id = _pending_create_risk(client)

    response = client.post("/reject", json={"approval_id": approval_id})

    assert response.status_code == 200
    assert response.json()["error_code"] == "APPROVAL_REJECTED"

    listed = client.post("/chat", json={"message": "Show all risks for PRJ-001"})
    assert "Scope creep" not in listed.json()["answer"]


def test_approve_without_risk_payload_reports_validation_error(client):
    approval_id = _pending_create_risk(client)

    response = client.post("/approve", json={"approval_id": approval_id})

    assert response.status_code == 200
    assert response.json()["error_code"] == "VALIDATION_ERROR"


def test_approve_with_a_blank_title_is_rejected_by_the_backend(client):
    """The form disables Approve on a blank title, but the server must not
    rely on that — a risk is never written without one."""
    approval_id = _pending_create_risk(client)

    response = client.post(
        "/approve",
        json={
            "approval_id": approval_id,
            "tool_input": {"risk_payload": {**RISK_PAYLOAD, "title": "   "}},
        },
    )

    assert response.json()["error_code"] == "VALIDATION_ERROR"


def test_approve_supplies_the_project_code_the_request_omitted(client):
    """A bare "add a risk" pends with no project; the form's project code
    field fills it in at approval time."""
    approval_id = client.post("/chat", json={"message": "add a risk"}).json()["approval_id"]

    response = client.post(
        "/approve",
        json={
            "approval_id": approval_id,
            "tool_input": {"project_code": "PRJ-001", "risk_payload": RISK_PAYLOAD},
        },
    )

    body = response.json()
    assert body["error_code"] is None
    assert "Scope creep" in body["answer"]


def test_approve_with_unknown_approval_id_returns_404(client):
    response = client.post("/approve", json={"approval_id": "does-not-exist"})

    assert response.status_code == 404


def test_reject_with_unknown_approval_id_returns_404(client):
    response = client.post("/reject", json={"approval_id": "does-not-exist"})

    assert response.status_code == 404


def test_reusing_a_resolved_approval_id_returns_404(client):
    approval_id = _pending_create_risk(client)
    client.post(
        "/approve",
        json={"approval_id": approval_id, "tool_input": {"risk_payload": RISK_PAYLOAD}},
    )

    second = client.post("/approve", json={"approval_id": approval_id})

    assert second.status_code == 404
