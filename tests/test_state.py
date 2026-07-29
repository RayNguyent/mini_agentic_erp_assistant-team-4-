import pytest
from pydantic import ValidationError

from app.errors import ErrorCode
from app.state import MAX_RETRIES, AgentState, NextAction


def test_valid_state_constructs():
    state = AgentState(intent="project_status", next_action=NextAction.ROUTE_DECISION)
    assert state.intent == "project_status"
    assert state.next_action == NextAction.ROUTE_DECISION
    assert state.retry_count == 0


def test_empty_intent_rejected():
    with pytest.raises(ValidationError, match="intent must not be empty"):
        AgentState(intent="   ", next_action=NextAction.ROUTE_DECISION)


def test_next_action_must_be_a_known_value():
    with pytest.raises(ValidationError):
        AgentState(intent="project_status", next_action="NOT_a_real_action")


@pytest.mark.parametrize("retry_count", [-1, MAX_RETRIES + 1])
def test_retry_count_out_of_bounds_rejected(retry_count):
    with pytest.raises(ValidationError):
        AgentState(
            intent="project_status",
            next_action=NextAction.ROUTE_DECISION,
            retry_count=retry_count,
        )


@pytest.mark.parametrize("retry_count", [0, MAX_RETRIES])
def test_retry_count_within_bounds_accepted(retry_count):
    state = AgentState(
        intent="project_status",
        next_action=NextAction.ROUTE_DECISION,
        retry_count=retry_count,
    )
    assert state.retry_count == retry_count


def test_tool_output_on_unapproved_write_action_rejected():
    with pytest.raises(ValidationError, match="approved is not True"):
        AgentState(
            intent="create_risk",
            next_action=NextAction.FORMAT_RESPONSE,
            approval_required=True,
            approved=False,
            tool_output={"id": "RISK-1"},
        )


def test_tool_output_on_approved_write_action_accepted():
    state = AgentState(
        intent="create_risk",
        next_action=NextAction.FORMAT_RESPONSE,
        approval_required=True,
        approved=True,
        tool_output={"id": "RISK-1"},
    )
    assert state.tool_output == {"id": "RISK-1"}


def test_pending_approval_without_tool_output_accepted():
    # approved=True but not yet executed must stay valid (approval precedes execution).
    state = AgentState(
        intent="create_risk",
        next_action=NextAction.EXECUTE_WRITE_TOOL,
        approval_required=True,
        approved=True,
        tool_output=None,
    )
    assert state.tool_output is None


def test_approved_rejected_when_approval_not_required():
    with pytest.raises(ValidationError, match="approved must not be set"):
        AgentState(
            intent="project_status",
            next_action=NextAction.DONE,
            approval_required=False,
            approved=False,
        )


def test_approved_none_accepted_when_approval_not_required():
    state = AgentState(
        intent="project_status",
        next_action=NextAction.DONE,
        approval_required=False,
        approved=None,
        tool_output={"name": "ERP Rollout"},
    )
    assert state.approved is None


def test_tool_output_and_error_code_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="must not both be set"):
        AgentState(
            intent="project_status",
            next_action=NextAction.FORMAT_RESPONSE,
            tool_output={"name": "ERP Rollout"},
            error_code=ErrorCode.TIMEOUT,
        )
