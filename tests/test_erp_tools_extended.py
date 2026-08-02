"""list_project_tasks, get_sprint_progress, get_budget_summary — against the
real MockERPProvider fixtures under data/, with a fixed evaluation date so
overdue/mapping-profile logic is deterministic."""

from datetime import date

import pytest

from app.errors import NotConfiguredError, NotFoundError, ToolValidationError
from app.providers.erp import MockERPProvider, is_overdue
from app.tools.erp_tools import (
    make_get_budget_summary,
    make_get_sprint_progress,
    make_list_project_tasks,
)

EVAL_DATE = date(2026, 8, 2)  # "today" per the session's real-world context
# Falls inside MS-001's window (2026-07-13 to 2026-07-24), used specifically
# to exercise "pick the milestone covering today" default-selection logic —
# by EVAL_DATE, MS-001 has already ended and MS-002 is the covering window,
# which is a realistic date-drift scenario but not what that one test targets.
MS_001_CURRENT_DATE = date(2026, 7, 20)


@pytest.fixture
def provider():
    return MockERPProvider()


# --- is_overdue: the shared rule --------------------------------------------


@pytest.mark.parametrize(
    "deadline,state,expected",
    [
        ("2026-07-01", "open", True),
        ("2026-07-01", "done", False),  # closed states are never overdue
        ("2026-07-01", "cancelled", False),
        ("2026-09-01", "open", False),  # future deadline
        (None, "open", False),  # no deadline at all
        ("not-a-date", "open", False),  # malformed input degrades safely
    ],
)
def test_is_overdue_rule(deadline, state, expected):
    assert is_overdue(deadline, state, EVAL_DATE) is expected


# --- list_project_tasks --------------------------------------------------


def test_list_project_tasks_returns_all_tasks_for_the_project(provider):
    tool = make_list_project_tasks(provider, today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001"})
    assert result["project_code"] == "PRJ-001"
    assert len(result["tasks"]) == 7  # matches data/tasks.json


def test_list_project_tasks_flags_overdue_open_tasks(provider):
    tool = make_list_project_tasks(provider, today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001", "status": "overdue"})
    ids = {t["task_id"] for t in result["tasks"]}
    assert "TASK-007" in ids  # open, deadline 2026-07-21, before EVAL_DATE
    assert all(t["overdue"] for t in result["tasks"])


def test_list_project_tasks_filters_by_exact_state():
    tool = make_list_project_tasks(MockERPProvider(), today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001", "status": "blocked"})
    assert {t["task_id"] for t in result["tasks"]} == {"TASK-004"}


def test_list_project_tasks_respects_the_limit():
    tool = make_list_project_tasks(MockERPProvider(), today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001", "limit": 2})
    assert len(result["tasks"]) == 2


def test_list_project_tasks_unknown_project_raises_not_found():
    tool = make_list_project_tasks(MockERPProvider(), today=EVAL_DATE)
    with pytest.raises(NotFoundError):
        tool({"project_code": "PRJ-999"})


def test_list_project_tasks_rejects_a_nonpositive_limit():
    tool = make_list_project_tasks(MockERPProvider(), today=EVAL_DATE)
    with pytest.raises(ToolValidationError):
        tool({"project_code": "PRJ-001", "limit": 0})


def test_a_task_with_no_milestone_or_deadline_is_never_overdue():
    tool = make_list_project_tasks(MockERPProvider(), today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001"})
    orphan = next(t for t in result["tasks"] if t["task_id"] == "TASK-006")
    assert orphan["milestone_id"] is None and orphan["overdue"] is False


# --- get_sprint_progress --------------------------------------------------


def test_sprint_progress_names_its_mapping_profile():
    tool = make_get_sprint_progress(MockERPProvider(), today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001"})
    assert result["mapping_profile"] == "milestone-as-iteration"


def test_sprint_progress_defaults_to_the_milestone_covering_today():
    tool = make_get_sprint_progress(MockERPProvider(), today=MS_001_CURRENT_DATE)
    result = tool({"project_code": "PRJ-001"})
    assert result["iteration_label"].startswith("Sprint 4")
    assert result["committed"] == 5  # MS-001 has 5 linked tasks


def test_sprint_progress_picks_the_nearest_milestone_once_the_current_one_has_ended():
    # By EVAL_DATE (2026-08-02) MS-001 has ended and MS-002 is the covering
    # window — "nearest" must not silently fall back to the first milestone.
    tool = make_get_sprint_progress(MockERPProvider(), today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001"})
    assert result["iteration_label"].startswith("Sprint 5")


def test_sprint_progress_computes_completion_and_overdue_counts_for_an_explicit_iteration():
    # By EVAL_DATE (2026-08-02), sprint 4 (2026-07-13 to 2026-07-24) is long
    # over, so every one of its still-open/in-progress/blocked tasks reads as
    # overdue — not just the one with the latest deadline.
    tool = make_get_sprint_progress(MockERPProvider(), today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001", "iteration_ref": "MS-001"})
    assert result["completed"] == 2  # TASK-001, TASK-002
    assert result["overdue"] == 3  # TASK-003, TASK-004, TASK-007
    assert result["completion_pct"] == pytest.approx(40.0)


def test_sprint_progress_accepts_an_explicit_iteration_ref():
    tool = make_get_sprint_progress(MockERPProvider(), today=EVAL_DATE)
    result = tool({"project_code": "PRJ-001", "iteration_ref": "MS-002"})
    assert result["iteration_label"].startswith("Sprint 5")


def test_sprint_progress_unknown_iteration_ref_is_not_found():
    tool = make_get_sprint_progress(MockERPProvider(), today=EVAL_DATE)
    with pytest.raises(NotFoundError):
        tool({"project_code": "PRJ-001", "iteration_ref": "MS-999"})


def test_sprint_progress_with_no_milestones_is_not_configured_not_a_guess():
    tool = make_get_sprint_progress(MockERPProvider(), today=EVAL_DATE)
    with pytest.raises(NotConfiguredError):
        tool({"project_code": "PRJ-002"})  # no milestones in the fixture


# --- get_budget_summary --------------------------------------------------


def test_budget_summary_computes_remaining_and_variance():
    tool = make_get_budget_summary(MockERPProvider())
    result = tool({"project_code": "PRJ-001"})
    assert result["remaining"] == pytest.approx(1200000 - 742000 - 188000)
    assert result["variance"] == pytest.approx(1200000 - 742000)


def test_budget_summary_reports_incomplete_data_rather_than_hiding_it():
    tool = make_get_budget_summary(MockERPProvider())
    result = tool({"project_code": "PRJ-001"})
    assert "expenses_final" in result["completeness_flags"]


def test_budget_summary_with_no_budget_record_is_not_configured():
    # PRJ-003 has no entry at all in data/budgets.json.
    tool = make_get_budget_summary(MockERPProvider())
    with pytest.raises(NotConfiguredError):
        tool({"project_code": "PRJ-003"})


def test_budget_summary_with_all_null_fields_is_not_configured_not_zero():
    # PRJ-002's budget record exists but every figure is null — this must
    # never render as a confident "$0 remaining".
    tool = make_get_budget_summary(MockERPProvider())
    result = tool({"project_code": "PRJ-002"})
    assert result["planned_budget"] is None
    assert result["remaining"] is None
    assert "planned_budget" in result["completeness_flags"]


def test_budget_summary_unknown_project_is_not_found_not_not_configured():
    tool = make_get_budget_summary(MockERPProvider())
    with pytest.raises(NotFoundError):
        tool({"project_code": "PRJ-999"})
