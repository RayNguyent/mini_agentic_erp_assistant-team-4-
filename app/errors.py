from enum import Enum


class ErrorCode(str, Enum):
    NOT_FOUND = "NOT_FOUND"
    TIMEOUT = "TIMEOUT"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    UNSUPPORTED_INTENT = "UNSUPPORTED_INTENT"
    APPROVAL_REJECTED = "APPROVAL_REJECTED"
    UNKNOWN = "UNKNOWN"

    # Spec-mandated normalized outcomes (final-project.pdf §9 "Odoo adapter
    # acceptance rules") — a provider port must collapse vendor-specific
    # failures onto this small typed set rather than leaking raw exceptions.
    FORBIDDEN = "FORBIDDEN"
    NOT_CONFIGURED = "NOT_CONFIGURED"
    NOT_SUPPORTED = "NOT_SUPPORTED"


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


class ForbiddenError(NonRetryableToolError):
    """The caller is identified but lacks the permission this action requires.
    Distinct from NotFoundError: the resource exists, the caller may not see it."""

    code = ErrorCode.FORBIDDEN


class NotConfiguredError(NonRetryableToolError):
    """The requested data exists in principle but this deployment has no
    source configured for it (e.g. no budget module wired up). Returned
    instead of a zero/empty value, which would look like a real answer."""

    code = ErrorCode.NOT_CONFIGURED


class NotSupportedError(NonRetryableToolError):
    """The requested capability is not available in the current provider
    profile (e.g. a risk register model not installed). Distinct from
    NOT_CONFIGURED: this is a capability gap, not a missing setting."""

    code = ErrorCode.NOT_SUPPORTED


class NonRetryableProviderError(NonRetryableToolError):
    """A provider call failed in a way that will not change on retry
    (e.g. bad credentials, malformed request)."""

    code = ErrorCode.PROVIDER_ERROR
