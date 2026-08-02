from typing import Literal

from pydantic import BaseModel, Field

from app.state import Citation


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

    # Route/trace metadata the browser UI surfaces (final-project.pdf's
    # "route/model/trace metadata" requirement) rather than hiding routing
    # decisions in free-form text.
    route: str | None = None
    agent_path: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    request_id: str | None = None
    provider: str | None = None
    model: str | None = None


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


class ReadinessResponse(BaseModel):
    status: Literal["ok", "degraded"]
    llm_provider: str
    model: str | None = None
    base_url: str | None = None
    credential_configured: bool
    tokenizer: str
    rag_index_chunks: int
    rag_vector_index: str
    rag_vector_count: int
    embeddings_enabled: bool
    erp_provider: str


class ToolInfo(BaseModel):
    name: str
    description: str
    permission: str
    side_effect: str
    timeout_s: float
    retry_limit: int
    audit_category: str


class ToolsResponse(BaseModel):
    tools: list[ToolInfo]
