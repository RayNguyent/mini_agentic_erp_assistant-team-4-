from pydantic import ValidationError

from app.errors import NotFoundError, ToolValidationError
from app.providers.erp import ERPProvider
from app.tools.schemas import (
    CreateRiskInput,
    ListRisksInput,
    ProjectStatusInput,
    ProjectStatusOutput,
    RiskOutput,
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
