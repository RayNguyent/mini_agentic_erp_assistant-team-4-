from app.runtime import resume, run
from app.state import NextAction
from app.tools.registry import build_default_registry


def test_project_status_scenario():
    registry = build_default_registry()
    state = run("What's the status of PRJ-001?", registry)
    assert state.next_action == NextAction.DONE
    assert state.error_code is None
    assert state.tool_output["project_code"] == "PRJ-001"


def test_list_risks_scenario():
    registry = build_default_registry()
    state = run("Show all risks for PRJ-001", registry)
    assert state.next_action == NextAction.DONE
    assert state.error_code is None
    assert len(state.tool_output["risks"]) == 2


def test_create_risk_scenario_requires_approval_then_executes():
    registry = build_default_registry()
    state = run("Create a risk for PRJ-002", registry)
    assert state.next_action == NextAction.AWAIT_APPROVAL

    state = state.model_copy(
        update={
            "approved": True,
            "tool_input": {
                **(state.tool_input or {}),
                "risk_payload": {"title": "Scope creep", "severity": "medium", "description": ""},
            },
        }
    )
    state = resume(state, registry)

    assert state.next_action == NextAction.DONE
    assert state.error_code is None
    assert state.tool_output["project_code"] == "PRJ-002"


def test_unsupported_request_scenario():
    registry = build_default_registry()
    state = run("What's the weather today?", registry)
    assert state.next_action == NextAction.DONE
    assert state.error_code == "UNSUPPORTED_INTENT"
