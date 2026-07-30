from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] | None = None


class RiskDraft(BaseModel):
    """Pre-fill for the create_risk approval form: whatever the classifier
    managed to extract, with blanks for whatever it didn't."""

    project_code: str = ""
    title: str = ""
    severity: Literal["low", "medium", "high"] = "medium"
    description: str = ""


class ChatResponse(BaseModel):
    answer: str
    tool_used: str | None = None
    approval_required: bool = False
    approval_id: str | None = None
    error_code: str | None = None
    risk_draft: RiskDraft | None = None


class ApproveRequest(BaseModel):
    approval_id: str
    # Merged into the pending state's tool_input before resuming — how a write
    # tool's structured arguments (e.g. create_risk's risk_payload) get supplied
    # when the deterministic classifier could only extract a project_code.
    tool_input: dict | None = None


class RejectRequest(BaseModel):
    approval_id: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
