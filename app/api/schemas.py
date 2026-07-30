from typing import Literal

from pydantic import BaseModel, Field


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    history: list[ChatTurn] | None = None


class ChatResponse(BaseModel):
    answer: str
    tool_used: str | None = None
    approval_required: bool = False
    approval_id: str | None = None
    error_code: str | None = None


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
