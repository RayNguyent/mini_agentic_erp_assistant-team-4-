from typing import Literal

from pydantic import BaseModel

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
    """New-risk fields supplied by the caller of create_risk."""

    title: str
    severity: RiskSeverity
    description: str = ""


class ListRisksInput(BaseModel):
    project_code: str


class CreateRiskInput(BaseModel):
    project_code: str
    risk_payload: RiskPayload


class RiskOutput(BaseModel):
    id: str
    project_code: str
    title: str
    severity: RiskSeverity
    status: str
    description: str = ""
    created_at: str
