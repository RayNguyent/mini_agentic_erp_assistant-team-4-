import pytest

from app.approvals.store import ApprovalNotFoundError, build_default_store
from app.state import AgentState, NextAction


def _pending_state() -> AgentState:
    return AgentState(
        intent="create_risk",
        selected_tool="create_risk",
        tool_input={"project_code": "PRJ-001"},
        approval_required=True,
        next_action=NextAction.AWAIT_APPROVAL,
    )


def test_create_then_pop_returns_the_same_state():
    store = build_default_store()
    state = _pending_state()

    approval_id = store.create(state)

    assert store.pop(approval_id) == state


def test_pop_unknown_id_raises():
    store = build_default_store()

    with pytest.raises(ApprovalNotFoundError):
        store.pop("does-not-exist")


def test_pop_twice_raises_on_second_call():
    store = build_default_store()
    approval_id = store.create(_pending_state())

    store.pop(approval_id)

    with pytest.raises(ApprovalNotFoundError):
        store.pop(approval_id)


def test_ids_are_unique_per_created_approval():
    store = build_default_store()

    first = store.create(_pending_state())
    second = store.create(_pending_state())

    assert first != second
