def test_project_status_scenario(client):
    response = client.post("/chat", json={"message": "What's the status of PRJ-001?"})

    assert response.status_code == 200
    body = response.json()
    assert body["tool_used"] == "get_project_status"
    assert body["error_code"] is None
    assert body["approval_required"] is False
    assert "ERP Platform Rollout" in body["answer"]


def test_list_risks_scenario(client):
    response = client.post("/chat", json={"message": "Show all risks for PRJ-001"})

    assert response.status_code == 200
    body = response.json()
    assert body["tool_used"] == "list_risks"
    assert body["error_code"] is None


def test_create_risk_scenario_requires_approval(client):
    response = client.post("/chat", json={"message": "Create a risk for PRJ-001"})

    assert response.status_code == 200
    body = response.json()
    assert body["tool_used"] == "create_risk"
    assert body["approval_required"] is True
    assert body["approval_id"]


def test_bare_create_risk_request_goes_straight_to_the_approval_form(client):
    """No project, title, or severity given — the user must still get the form
    to fill in rather than a chat reply asking for those details."""
    response = client.post("/chat", json={"message": "add a risk"})

    body = response.json()
    assert body["tool_used"] == "create_risk"
    assert body["approval_required"] is True
    assert body["risk_draft"] == {
        "project_code": "",
        "title": "",
        "severity": "medium",
        "description": "",
    }


def test_create_risk_draft_prefills_what_the_user_already_said(client):
    response = client.post("/chat", json={"message": "Create a risk for PRJ-001"})

    assert response.json()["risk_draft"]["project_code"] == "PRJ-001"


def test_read_only_requests_carry_no_risk_draft(client):
    response = client.post("/chat", json={"message": "Show all risks for PRJ-001"})

    assert response.json()["risk_draft"] is None


def test_unsupported_request_scenario_is_a_clean_refusal(client):
    response = client.post("/chat", json={"message": "What's the weather today?"})

    assert response.status_code == 200
    body = response.json()
    assert body["error_code"] == "UNSUPPORTED_INTENT"
    assert body["tool_used"] is None
    assert body["approval_required"] is False


def test_unknown_project_reports_not_found_without_failing_the_request(client):
    response = client.post("/chat", json={"message": "What's the status of PRJ-999?"})

    assert response.status_code == 200
    assert response.json()["error_code"] == "NOT_FOUND"


def test_empty_message_is_rejected_as_a_bad_request(client):
    response = client.post("/chat", json={"message": ""})

    assert response.status_code == 422
