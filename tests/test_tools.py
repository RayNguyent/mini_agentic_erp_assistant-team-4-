import pytest

from app.errors import NotFoundError, ToolValidationError
from app.providers.erp import MockERPProvider
from app.tools.registry import ToolRegistry


@pytest.fixture
def registry():
    return ToolRegistry(MockERPProvider())


# --- get_project_status ------------------------------------------------------


def test_get_project_status_returns_expected_shape(registry):
    tool = registry.get("get_project_status")
    result = tool({"project_code": "PRJ-001"})
    assert result["project_code"] == "PRJ-001"
    assert result["stage"] and result["owner"] and result["status_summary"]


def test_get_project_status_raises_not_found_for_unknown_project(registry):
    tool = registry.get("get_project_status")
    with pytest.raises(NotFoundError):
        tool({"project_code": "PRJ-999"})


def test_get_project_status_raises_validation_error_for_missing_project_code(registry):
    tool = registry.get("get_project_status")
    with pytest.raises(ToolValidationError):
        tool({})


# --- list_risks ---------------------------------------------------------------


def test_list_risks_returns_all_risks_for_project(registry):
    tool = registry.get("list_risks")
    result = tool({"project_code": "PRJ-001"})
    assert result["project_code"] == "PRJ-001"
    assert len(result["risks"]) == 2


def test_list_risks_returns_empty_list_for_project_with_no_risks(registry):
    tool = registry.get("list_risks")
    result = tool({"project_code": "PRJ-002"})
    assert result["risks"] == []


def test_list_risks_raises_not_found_for_unknown_project(registry):
    tool = registry.get("list_risks")
    with pytest.raises(NotFoundError):
        tool({"project_code": "PRJ-999"})


# --- create_risk ----------------------------------------------------------------


def test_create_risk_adds_new_risk_and_returns_it(registry):
    tool = registry.get("create_risk")
    result = tool(
        {
            "project_code": "PRJ-002",
            "risk_payload": {
                "title": "Integration delay",
                "severity": "medium",
                "description": "Vendor slipped milestone.",
            },
        }
    )
    assert result["project_code"] == "PRJ-002"
    assert result["status"] == "Open"
    assert result["title"] == "Integration delay"

    # Confirms the write actually landed, not just an echo.
    listed = registry.get("list_risks")({"project_code": "PRJ-002"})
    assert any(r["id"] == result["id"] for r in listed["risks"])


def test_create_risk_requires_risk_payload(registry):
    tool = registry.get("create_risk")
    with pytest.raises(ToolValidationError):
        tool({"project_code": "PRJ-002"})


def test_create_risk_raises_not_found_for_unknown_project(registry):
    tool = registry.get("create_risk")
    with pytest.raises(NotFoundError):
        tool(
            {
                "project_code": "PRJ-999",
                "risk_payload": {"title": "x", "severity": "low", "description": ""},
            }
        )


def test_create_risk_raises_validation_error_for_bad_severity(registry):
    tool = registry.get("create_risk")
    with pytest.raises(ToolValidationError):
        tool(
            {
                "project_code": "PRJ-002",
                "risk_payload": {"title": "x", "severity": "extreme", "description": ""},
            }
        )
