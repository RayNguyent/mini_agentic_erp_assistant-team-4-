from typing import Literal

from pydantic import BaseModel, Field, field_validator

RiskSeverity = Literal["low", "medium", "high"]


class ProjectStatusInput(BaseModel):
    project_code: str


class ProjectStatusOutput(BaseModel):
    project_code: str
    name: str
    stage: str
    owner: str
    status_summary: str


class RiskPayload(BaseModel):
    """New-risk fields supplied by the caller of create_risk.

    Enforced at execution time only — a risk is never written with a blank
    title, no matter what the approval form submits.
    """

    title: str
    severity: RiskSeverity
    description: str = ""

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("title must not be blank")
        return value.strip()


class RiskDraftPayload(BaseModel):
    """A partially-filled RiskPayload, as much as the classifier could pull
    out of the user's message. Every field is optional so an under-specified
    request still routes to create_risk; the approval form collects the rest.
    """

    title: str | None = None
    severity: RiskSeverity | None = None
    description: str = ""


class ListRisksInput(BaseModel):
    project_code: str


class CreateRiskInput(BaseModel):
    project_code: str
    risk_payload: RiskPayload


class CreateRiskDraftInput(BaseModel):
    """Classification-time shape of create_risk, handed to the LLM instead of
    CreateRiskInput. Nothing is required: a bare "add a risk" must still
    produce a tool call so the user gets the approval form to fill in, rather
    than the model declining and falling back to a chat reply asking for
    details. CreateRiskInput is what actually gates execution.
    """

    project_code: str | None = None
    risk_payload: RiskDraftPayload | None = None


class RiskOutput(BaseModel):
    id: str
    project_code: str
    title: str
    severity: RiskSeverity
    status: str
    description: str = ""
    created_at: str


# --- list_project_tasks -------------------------------------------------------

TaskStatusFilter = Literal["open", "in_progress", "blocked", "done", "cancelled", "overdue"]


class ListProjectTasksInput(BaseModel):
    project_code: str
    status: TaskStatusFilter | None = None
    limit: int | None = None

    @field_validator("limit")
    @classmethod
    def limit_must_be_positive(cls, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError("limit must be a positive integer")
        return value


class TaskOutput(BaseModel):
    task_id: str
    project_code: str
    title: str
    state: str
    milestone_id: str | None = None
    deadline: str | None = None
    overdue: bool = False


# --- get_sprint_progress -------------------------------------------------------


class SprintProgressInput(BaseModel):
    project_code: str
    iteration_ref: str | None = None


class SprintProgressOutput(BaseModel):
    project_code: str
    iteration_label: str
    start_date: str
    end_date: str
    # Named explicitly per final-project.pdf: "the response must identify the
    # active mapping profile; never label all project tasks as one sprint
    # silently." This corpus's default profile maps a milestone to a sprint.
    mapping_profile: str = "milestone-as-iteration"
    committed: int
    completed: int
    open: int
    overdue: int
    completion_pct: float
    task_ids: list[str] = Field(default_factory=list)


# --- get_budget_summary -------------------------------------------------------


class BudgetSummaryInput(BaseModel):
    project_code: str


class BudgetSummaryOutput(BaseModel):
    project_code: str
    currency: str
    planned_budget: float | None
    actual_cost: float | None
    committed_cost: float | None
    forecast_revenue: float | None
    remaining: float | None
    variance: float | None
    as_of: str
    completeness_flags: list[str] = Field(default_factory=list)
