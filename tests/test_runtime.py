import pytest

from app.errors import NotFoundError, ToolTimeoutError
from app.runtime import execute_write_tool, parse_intent, resume, route_decision, run
from app.state import NextAction


class FakeRegistry:
    def __init__(self, tools: dict):
        self._tools = tools

    def get(self, name):
        return self._tools.get(name)


# --- parse_intent / route_decision -----------------------------------------


def test_parse_intent_extracts_project_code():
    state = parse_intent("What's the status of PRJ-001?")
    assert state.intent == "project_status"
    assert state.tool_input == {"project_code": "PRJ-001"}
    assert state.next_action == NextAction.ROUTE_DECISION


def test_route_decision_selects_read_tool():
    state = parse_intent("Show all risks for PRJ-001")
    state = route_decision(state)
    assert state.selected_tool == "list_risks"
    assert state.next_action == NextAction.EXECUTE_READ_TOOL
    assert state.approval_required is False


def test_route_decision_selects_write_tool_and_requires_approval():
    state = parse_intent("Create a risk for PRJ-001")
    state = route_decision(state)
    assert state.selected_tool == "create_risk"
    assert state.next_action == NextAction.EXECUTE_WRITE_TOOL
    assert state.approval_required is True


def test_route_decision_rejects_unsupported_intent():
    state = parse_intent("What's the weather today?")
    state = route_decision(state)
    assert state.next_action == NextAction.FORMAT_RESPONSE
    assert state.error_code == "UNSUPPORTED_INTENT"
    assert state.tool_output is None


# --- Scenario 1: read tool success ------------------------------------------


def test_run_read_tool_success():
    registry = FakeRegistry(
        {"get_project_status": lambda inp: {"project_code": inp["project_code"], "stage": "Build"}}
    )
    state = run("What's the status of PRJ-001?", registry)
    assert state.next_action == NextAction.DONE
    assert state.selected_tool == "get_project_status"
    assert state.error_code is None
    assert state.answer is not None


# --- Scenario 2: write tool, approval flow ----------------------------------


def test_write_tool_halts_pending_approval():
    registry = FakeRegistry({"create_risk": lambda inp: {"id": "RISK-1"}})
    state = run("Create a risk for PRJ-001", registry)
    assert state.next_action == NextAction.AWAIT_APPROVAL
    assert state.approval_required is True
    assert state.approved is None


def test_write_tool_approved_then_resumed_executes():
    calls = []
    registry = FakeRegistry(
        {"create_risk": lambda inp: calls.append(inp) or {"id": "RISK-1", **inp}}
    )
    state = run("Create a risk for PRJ-001", registry)
    state = state.model_copy(update={"approved": True})

    state = resume(state, registry)

    assert state.next_action == NextAction.DONE
    assert state.tool_output == {"id": "RISK-1", "project_code": "PRJ-001"}
    assert len(calls) == 1


def test_write_tool_rejected_is_not_executed():
    calls = []
    registry = FakeRegistry({"create_risk": lambda inp: calls.append(inp) or {"id": "RISK-1"}})
    state = run("Create a risk for PRJ-002", registry)
    state = state.model_copy(update={"approved": False})

    state = resume(state, registry)

    assert state.next_action == NextAction.DONE
    assert state.tool_output is None
    assert state.error_code == "APPROVAL_REJECTED"
    assert len(calls) == 0


def test_resume_raises_if_not_awaiting_approval():
    registry = FakeRegistry({"create_risk": lambda inp: {"id": "RISK-1"}})
    state = run("What's the status of PRJ-001?", FakeRegistry({"get_project_status": lambda inp: {}}))
    with pytest.raises(ValueError, match="not awaiting approval"):
        resume(state, registry)


# --- Scenario 4: unsupported request -----------------------------------------


def test_run_unsupported_request_is_clean_refusal():
    registry = FakeRegistry({})
    state = run("What's the weather today?", registry)
    assert state.next_action == NextAction.DONE
    assert state.error_code == "UNSUPPORTED_INTENT"
    assert state.tool_output is None
    assert state.answer


# --- Retry logic --------------------------------------------------------------


def test_retryable_error_retries_up_to_max_then_fails():
    attempts = []

    def flaky(inp):
        attempts.append(inp)
        raise ToolTimeoutError("upstream timed out")

    registry = FakeRegistry({"get_project_status": flaky})
    state = run("status of PRJ-999", registry)

    assert len(attempts) == 3  # initial attempt + 2 retries
    assert state.next_action == NextAction.DONE
    assert state.error_code == "TIMEOUT"
    assert state.retry_count == 2


def test_non_retryable_error_fails_immediately_without_retry():
    attempts = []

    def not_found(inp):
        attempts.append(inp)
        raise NotFoundError("PRJ-000 not found")

    registry = FakeRegistry({"get_project_status": not_found})
    state = run("status of PRJ-000", registry)

    assert len(attempts) == 1
    assert state.error_code == "NOT_FOUND"
    assert state.retry_count == 0


def test_missing_tool_registration_reports_not_found():
    registry = FakeRegistry({})
    state = execute_write_tool(
        route_decision(parse_intent("Create a risk for PRJ-001")).model_copy(
            update={"approved": True}
        ),
        registry,
    )
    assert state.error_code == "NOT_FOUND"
