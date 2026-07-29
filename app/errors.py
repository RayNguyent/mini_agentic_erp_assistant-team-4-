from enum import Enum


class ErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_INTENT = "UNSUPPORTED_INTENT"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    UNKNOWN = "UNKNOWN"


class ToolError(Exception):
    """Base class for all tool execution errors."""

    code: ErrorCode = ErrorCode.UNKNOWN

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


class RetryableToolError(ToolError):
    """Transient failure — worth retrying (up to MAX_RETRIES)."""


class ToolTimeoutError(RetryableToolError):
    code = ErrorCode.TIMEOUT


class ProviderError(RetryableToolError):
    code = ErrorCode.PROVIDER_ERROR


class NonRetryableToolError(ToolError):
    """Deterministic failure — retrying will not change the outcome."""


class NotFoundError(NonRetryableToolError):
    code = ErrorCode.NOT_FOUND


class ToolValidationError(NonRetryableToolError):
    code = ErrorCode.VALIDATION_ERROR
