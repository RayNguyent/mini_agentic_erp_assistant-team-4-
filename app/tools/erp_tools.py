from datetime import date, datetime, timezone

from pydantic import ValidationError

from app.errors import NotConfiguredError, NotFoundError, ToolValidationError
from app.providers.erp import ERPProvider, is_overdue, require_budget_source
from app.tools.schemas import (
    BudgetSummaryInput,
    BudgetSummaryOutput,
    CreateRiskInput,
    ListProjectTasksInput,
    ListRisksInput,
    ProjectStatusInput,
    ProjectStatusOutput,
    RiskOutput,
    SprintProgressInput,
    SprintProgressOutput,
    TaskOutput,
)


def _as_validation_error(exc: ValidationError) -> ToolValidationError:
    return ToolValidationError(str(exc))


def make_get_project_status(provider: ERPProvider):
    def get_project_status(tool_input: dict) -> dict:
        try:
            parsed = ProjectStatusInput(**tool_input)
        except ValidationError as exc:
            raise _as_validation_error(exc) from exc

        project = provider.get_project(parsed.project_code)
        if project is None:
            projects = provider.list_projects()
            available = ", ".join([p["project_code"] for p in projects])
            raise NotFoundError(
                f"Project {parsed.project_code} not found. Available projects: {available}"
            )

        return ProjectStatusOutput(**project).model_dump()

    return get_project_status


def make_list_risks(provider: ERPProvider):
    def list_risks(tool_input: dict) -> dict:
        try:
            parsed = ListRisksInput(**tool_input)
        except ValidationError as exc:
            raise _as_validation_error(exc) from exc

        if provider.get_project(parsed.project_code) is None:
            projects = provider.list_projects()
            available = ", ".join([p["project_code"] for p in projects])
            raise NotFoundError(
                f"Project {parsed.project_code} not found. Available projects: {available}"
            )

        risks = provider.list_risks(parsed.project_code)
        return {
            "project_code": parsed.project_code,
            "risks": [RiskOutput(**risk).model_dump() for risk in risks],
        }

    return list_risks


def make_create_risk(provider: ERPProvider):
    def create_risk(tool_input: dict) -> dict:
        try:
            parsed = CreateRiskInput(**tool_input)
        except ValidationError as exc:
            raise _as_validation_error(exc) from exc

        if provider.get_project(parsed.project_code) is None:
            projects = provider.list_projects()
            available = ", ".join([p["project_code"] for p in projects])
            raise NotFoundError(
                f"Project {parsed.project_code} not found. Available projects: {available}"
            )

        risk = provider.create_risk(parsed.project_code, parsed.risk_payload.model_dump())
        return RiskOutput(**risk).model_dump()

    return create_risk


def _require_project(provider: ERPProvider, project_code: str) -> None:
    if provider.get_project(project_code) is not None:
        return
    available = ", ".join(p["project_code"] for p in provider.list_projects())
    raise NotFoundError(f"Project {project_code} not found. Available projects: {available}")


def make_list_project_tasks(provider: ERPProvider, *, today: date | None = None):
    """`today` is injectable so overdue is testable deterministically; the
    real tool call always evaluates against the current date."""

    def list_project_tasks(tool_input: dict) -> dict:
        try:
            parsed = ListProjectTasksInput(**tool_input)
        except ValidationError as exc:
            raise _as_validation_error(exc) from exc

        _require_project(provider, parsed.project_code)
        as_of = today or datetime.now(timezone.utc).date()

        tasks = [
            TaskOutput(
                task_id=t["task_id"],
                project_code=t["project_code"],
                title=t["title"],
                state=t["state"],
                milestone_id=t.get("milestone_id"),
                deadline=t.get("deadline"),
                overdue=is_overdue(t.get("deadline"), t["state"], as_of),
            )
            for t in provider.list_tasks(parsed.project_code)
        ]

        if parsed.status == "overdue":
            tasks = [t for t in tasks if t.overdue]
        elif parsed.status is not None:
            tasks = [t for t in tasks if t.state == parsed.status]

        if parsed.limit is not None:
            tasks = tasks[: parsed.limit]

        return {
            "project_code": parsed.project_code,
            "tasks": [t.model_dump() for t in tasks],
        }

    return list_project_tasks


def make_get_sprint_progress(provider: ERPProvider, *, today: date | None = None):
    def get_sprint_progress(tool_input: dict) -> dict:
        try:
            parsed = SprintProgressInput(**tool_input)
        except ValidationError as exc:
            raise _as_validation_error(exc) from exc

        _require_project(provider, parsed.project_code)
        as_of = today or datetime.now(timezone.utc).date()

        milestones = provider.list_milestones(parsed.project_code)
        if not milestones:
            raise NotConfiguredError(
                f"No milestone (delivery-iteration) data is configured for "
                f"{parsed.project_code} under the default mapping profile "
                "('milestone-as-iteration')."
            )

        milestone = None
        if parsed.iteration_ref:
            milestone = next((m for m in milestones if m["milestone_id"] == parsed.iteration_ref), None)
            if milestone is None:
                raise NotFoundError(
                    f"Iteration '{parsed.iteration_ref}' not found for {parsed.project_code}."
                )
        else:
            # Default: the milestone whose window contains today, else the
            # nearest upcoming one. Never silently pick "the first milestone" —
            # that is exactly the "labelling all tasks as one sprint" failure
            # the spec calls out.
            current = [m for m in milestones if m["start_date"] <= as_of.isoformat() <= m["end_date"]]
            milestone = current[0] if current else min(
                milestones, key=lambda m: abs(date.fromisoformat(m["start_date"]) - as_of)
            )

        tasks = [t for t in provider.list_tasks(parsed.project_code) if t.get("milestone_id") == milestone["milestone_id"]]
        completed = [t for t in tasks if t["state"] == "done"]
        overdue = [t for t in tasks if is_overdue(t.get("deadline"), t["state"], as_of)]
        openn = [t for t in tasks if t["state"] not in ("done", "cancelled")]

        completion_pct = round(100 * len(completed) / len(tasks), 1) if tasks else 0.0

        output = SprintProgressOutput(
            project_code=parsed.project_code,
            iteration_label=milestone["label"],
            start_date=milestone["start_date"],
            end_date=milestone["end_date"],
            committed=len(tasks),
            completed=len(completed),
            open=len(openn),
            overdue=len(overdue),
            completion_pct=completion_pct,
            task_ids=[t["task_id"] for t in tasks],
        )
        return output.model_dump()

    return get_sprint_progress


def make_get_budget_summary(provider: ERPProvider):
    def get_budget_summary(tool_input: dict) -> dict:
        try:
            parsed = BudgetSummaryInput(**tool_input)
        except ValidationError as exc:
            raise _as_validation_error(exc) from exc

        _require_project(provider, parsed.project_code)
        budget = require_budget_source(parsed.project_code, provider.get_budget(parsed.project_code))

        planned = budget.get("planned_budget")
        actual = budget.get("actual_cost")
        committed = budget.get("committed_cost")
        remaining = (planned - actual - committed) if None not in (planned, actual, committed) else None
        variance = (planned - actual) if None not in (planned, actual) else None

        output = BudgetSummaryOutput(
            project_code=parsed.project_code,
            currency=budget.get("currency", "USD"),
            planned_budget=planned,
            actual_cost=actual,
            committed_cost=committed,
            forecast_revenue=budget.get("forecast_revenue"),
            remaining=remaining,
            variance=variance,
            as_of=budget.get("as_of", ""),
            completeness_flags=list(budget.get("missing_sources") or []),
        )
        return output.model_dump()

    return get_budget_summary
