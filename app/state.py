from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from app.errors import ErrorCode

MAX_RETRIES = 2


class NextAction(str, Enum):
    PARSE_INTENT = "parse_intent"
    ROUTE_DECISION = "route_decision"
    EXECUTE_READ_TOOL = "execute_read_tool"
    EXECUTE_WRITE_TOOL = "execute_write_tool"
    GENERATE_RESPONSE = "generate_response"
    AWAIT_APPROVAL = "await_approval"
    FORMAT_RESPONSE = "format_response"
    DONE = "done"

    # Multi-agent nodes (app/agents). SUPERVISE plans which specialists run;
    # RUN_AGENTS executes them; SYNTHESIZE merges their results into one answer.
    SUPERVISE = "supervise"
    RUN_AGENTS = "run_agents"
    SYNTHESIZE = "synthesize"
    REFUSE = "refuse"


class IntentType(str, Enum):
    """The closed set of intents any classifier (rule-based or LLM-backed)
    may produce. Mirrors app.runtime.READ_TOOLS / WRITE_TOOLS keys."""

    PROJECT_STATUS = "project_status"
    LIST_RISKS = "list_risks"
    CREATE_RISK = "create_risk"
    LIST_PROJECT_TASKS = "list_project_tasks"
    SPRINT_PROGRESS = "sprint_progress"
    BUDGET_SUMMARY = "budget_summary"
    DOC_QUESTION = "doc_question"
    UNSUPPORTED = "unsupported"


class Route(str, Enum):
    """The top-level path a request took. Recorded in state and in the trace so
    a reviewer can see *why* an answer looks the way it does without reading
    the prose."""

    TOOL = "tool"
    RAG = "rag"
    MULTI_AGENT = "multi_agent"
    CONVERSATION = "conversation"
    REFUSAL = "refusal"


RiskLevel = Literal["none", "low", "medium", "high"]

Role = Literal["developer", "project_manager", "auditor"]


class Actor(BaseModel):
    """Who is making the request. Every request carries one; tool permission
    checks and document ACL filtering both key off `role`."""

    user_id: str
    role: Role

    model_config = {"frozen": True}


class Citation(BaseModel):
    """A single grounded reference. `chunk_id` is what the answer must cite;
    the rest is what the UI renders on a source card."""

    doc_id: str
    chunk_id: str
    title: str
    heading: str = ""
    snippet: str = ""
    score: float = 0.0
    classification: str = "public"

    model_config = {"frozen": True}


class AgentStep(BaseModel):
    """One entry in the supervisor's typed plan."""

    agent: str
    rationale: str = ""
    inputs: dict = Field(default_factory=dict)


class ContextItem(BaseModel):
    """One candidate block considered by the context builder."""

    kind: str
    ref: str
    tokens: int
    included: bool
    reason: str = ""


class ContextBuildLog(BaseModel):
    """Required evidence: what went into the prompt and what was left out.

    The `excluded` half is the point — a budget that silently drops the safety
    policy is indistinguishable from one that never had it.
    """

    budget_tokens: int = 0
    used_tokens: int = 0
    items: list[ContextItem] = Field(default_factory=list)

    @property
    def excluded(self) -> list[ContextItem]:
        return [item for item in self.items if not item.included]


class AgentResult(BaseModel):
    """What a specialist agent hands back to the synthesizer."""

    agent: str
    ok: bool = True
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    tool_used: str | None = None
    tool_output: dict | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None
    elapsed_ms: float = 0.0
    retries: int = 0
    # Set by DocResearcher: what was admitted into the evidence prompt under
    # the token budget, and what was excluded and why.
    context_log: ContextBuildLog | None = None
    # Set by RiskWriter when create_risk needs a human decision before it can
    # run. Explicit rather than inferred from "no output, no error" so the
    # supervisor's dispatch loop can branch on it unambiguously.
    awaiting_approval: bool = False


class AgentState(BaseModel):
    intent: str
    selected_tool: str | None = None
    tool_input: dict | None = None
    tool_output: dict | None = None
    approval_required: bool = False
    approved: bool | None = None
    retry_count: int = 0
    next_action: NextAction
    error_code: ErrorCode | None = None
    error_message: str | None = None
    answer: str | None = None
    status: str = "Analyzing your request..."

    # --- Request identity and routing ------------------------------------
    request_id: str | None = None
    actor: Actor | None = None
    route: Route | None = None
    risk_level: RiskLevel = "none"

    # --- Grounding --------------------------------------------------------
    required_context: list[str] = Field(default_factory=list)
    retrieved_sources: list[Citation] = Field(default_factory=list)
    refused: bool = False

    # --- Multi-agent ------------------------------------------------------
    plan: list[AgentStep] = Field(default_factory=list)
    agent_results: dict[str, AgentResult] = Field(default_factory=dict)

    # --- Context + write safety ------------------------------------------
    context_log: ContextBuildLog | None = None
    idempotency_key: str | None = None

    @field_validator("intent")
    @classmethod
    def intent_must_not_be_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("intent must not be empty")
        return value

    @field_validator("retry_count")
    @classmethod
    def retry_count_within_bounds(cls, value: int) -> int:
        if value < 0:
            raise ValueError("retry_count must not be negative")
        if value > MAX_RETRIES:
            raise ValueError(f"retry_count must not exceed MAX_RETRIES ({MAX_RETRIES})")
        return value

    @model_validator(mode="after")
    def tool_output_requires_approval(self) -> "AgentState":
        # A write tool's result must never exist without a recorded approval.
        # Note: this does NOT require the reverse (approved=True doesn't need
        # tool_output yet) since approval happens before execution in the
        # runtime flow, and that in-flight state must stay valid.
        if self.tool_output is not None and self.approval_required and not self.approved:
            raise ValueError(
                "tool_output is set on an approval-required action but approved is not True"
            )
        return self

    @model_validator(mode="after")
    def output_and_error_are_mutually_exclusive(self) -> "AgentState":
        if self.tool_output is not None and self.error_code is not None:
            raise ValueError("tool_output and error_code must not both be set")
        return self

    @model_validator(mode="after")
    def approved_only_meaningful_when_approval_required(self) -> "AgentState":
        # approved is set by the approval gate a write tool goes through; a
        # read tool (approval_required=False) never enters that gate, so
        # approved must stay None for it.
        if self.approved is not None and not self.approval_required:
            raise ValueError("approved must not be set unless approval_required is True")
        return self

    @model_validator(mode="after")
    def document_answers_must_cite(self) -> "AgentState":
        # The core RAG gate, enforced structurally rather than by prompt: if a
        # finished answer came from the document route and is not a refusal or
        # an error, it must carry at least one citation. An uncited grounded
        # answer is unrepresentable.
        answered = self.next_action == NextAction.DONE and bool(self.answer)
        if (
            self.route == Route.RAG
            and answered
            and not self.refused
            and self.error_code is None
            and not self.retrieved_sources
        ):
            raise ValueError(
                "a document-route answer must cite at least one retrieved source "
                "(set refused=True to return an ungrounded refusal instead)"
            )
        return self
