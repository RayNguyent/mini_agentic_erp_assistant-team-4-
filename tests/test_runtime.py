import pytest

from app.errors import NotFoundError, ToolTimeoutError
from app.providers.intent_classifier import make_llm_classifier
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


@pytest.mark.parametrize("message", ["status of PRJ 001", "status of PRJ_001", "status of prj001"])
def test_parse_intent_normalizes_project_code_format_drift(message):
    state = parse_intent(message)
    assert state.tool_input == {"project_code": "PRJ-001"}


def test_run_normalizes_project_code_before_calling_tool():
    calls = []
    registry = FakeRegistry(
        {"get_project_status": lambda inp: calls.append(inp) or {"project_code": inp["project_code"]}}
    )
    run("status of PRJ 001", registry)
    assert calls == [{"project_code": "PRJ-001"}]


# --- Carry-forward intent from history when the classifier declines --------


def test_parse_intent_carries_forward_read_intent_from_history_on_bare_followup():
    history = [
        {"role": "user", "content": "what is the status of prj 001"},
        {"role": "assistant", "content": "Project PRJ-001 — ERP Platform Rollout"},
    ]

    def declining_classify(message, history=None):
        return "unsupported", {}

    state = parse_intent("how about prj 002", classify=declining_classify, history=history)

    assert state.intent == "project_status"
    assert state.tool_input == {"project_code": "PRJ-002"}


def test_parse_intent_does_not_carry_forward_create_risk():
    history = [{"role": "user", "content": "create a risk for prj 001"}]

    def declining_classify(message, history=None):
        return "unsupported", {}

    state = parse_intent("how about prj 002", classify=declining_classify, history=history)

    assert state.intent == "unsupported"


def test_parse_intent_does_not_carry_forward_without_a_project_code_in_the_new_message():
    history = [{"role": "user", "content": "what is the status of prj 001"}]

    def declining_classify(message, history=None):
        return "unsupported", {}

    state = parse_intent("what about it", classify=declining_classify, history=history)

    assert state.intent == "unsupported"


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
    assert state.next_action == NextAction.GENERATE_RESPONSE
    assert state.error_code is None
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


def test_project_status_answer_is_rendered_as_bullets():
    registry = FakeRegistry(
        {
            "get_project_status": lambda inp: {
                "project_code": "PRJ-001",
                "name": "ERP Platform Rollout",
                "stage": "Build",
                "owner": "Alice Tran",
                "status_summary": "On track.",
            }
        }
    )
    state = run("What's the status of PRJ-001?", registry)
    assert state.answer == (
        "Project PRJ-001 — ERP Platform Rollout\n"
        "• Stage: Build\n"
        "• Owner: Alice Tran\n"
        "• Status: On track."
    )


def test_list_risks_answer_is_rendered_as_bullets():
    registry = FakeRegistry(
        {
            "list_risks": lambda inp: {
                "project_code": "PRJ-001",
                "risks": [
                    {
                        "id": "RISK-1",
                        "title": "Scope creep",
                        "severity": "medium",
                        "status": "open",
                    }
                ],
            }
        }
    )
    state = run("Show all risks for PRJ-001", registry)
    assert state.answer == (
        "Risks for PRJ-001:\n• [RISK-1] Scope creep (medium) — open"
    )


def test_list_risks_answer_reports_when_empty():
    registry = FakeRegistry(
        {"list_risks": lambda inp: {"project_code": "PRJ-001", "risks": []}}
    )
    state = run("Show all risks for PRJ-001", registry)
    assert state.answer == "No risks recorded for PRJ-001."


def test_create_risk_answer_is_rendered_as_bullets():
    calls = []
    registry = FakeRegistry(
        {
            "create_risk": lambda inp: calls.append(inp)
            or {
                "id": "RISK-1",
                "project_code": "PRJ-001",
                "title": "Scope creep",
                "severity": "medium",
                "status": "open",
            }
        }
    )
    state = run("Create a risk for PRJ-001", registry)
    state = state.model_copy(update={"approved": True})
    state = resume(state, registry)
    assert state.answer == (
        "Risk created for PRJ-001:\n"
        "• Title: Scope creep\n"
        "• Severity: medium\n"
        "• Status: open"
    )


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
    assert state.error_code is None
    assert state.tool_output is None
    assert state.answer


# --- LLM-backed classifier wiring -------------------------------------------


class _FakeLLMProvider:
    def __init__(self, response: str):
        self._response = response

    def generate(self, prompt, *, system=None, history=None):
        return self._response


def test_run_with_llm_backed_classifier_routes_like_the_default_one():
    provider = _FakeLLMProvider('{"intent": "project_status", "project_code": "PRJ-001"}')
    classify = make_llm_classifier(provider)
    registry = FakeRegistry(
        {"get_project_status": lambda inp: {"project_code": inp["project_code"], "stage": "Build"}}
    )

    state = run("Tell me about PRJ-001", registry, classify=classify)

    assert state.next_action == NextAction.DONE
    assert state.selected_tool == "get_project_status"
    assert state.tool_output == {"project_code": "PRJ-001", "stage": "Build"}


def test_run_with_llm_backed_classifier_falls_back_when_provider_misbehaves():
    provider = _FakeLLMProvider("not valid json")
    classify = make_llm_classifier(provider)
    registry = FakeRegistry({"get_project_status": lambda inp: {"stage": "Build"}})

    state = run("What's the status of PRJ-001?", registry, classify=classify)

    assert state.next_action == NextAction.DONE
    assert state.selected_tool == "get_project_status"


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
