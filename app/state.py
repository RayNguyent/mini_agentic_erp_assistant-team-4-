from enum import Enum

from pydantic import BaseModel, field_validator, model_validator

from app.errors import ErrorCode

MAX_RETRIES = 2


class NextAction(str, Enum):
    ROUTE_DECISION = "route_decision"
    EXECUTE_READ_TOOL = "execute_read_tool"
    EXECUTE_WRITE_TOOL = "execute_write_tool"
    AWAIT_APPROVAL = "await_approval"
    FORMAT_RESPONSE = "format_response"
    DONE = "done"


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
