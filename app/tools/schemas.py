from typing import Literal

from pydantic import BaseModel, field_validator

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
