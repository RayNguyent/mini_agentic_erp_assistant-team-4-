"""Guardrails and auth: authentication, role-based permissions, deterministic
prompt-injection screening, and the append-only audit log.

Enforcement lives at the boundaries that already exist (the tool gateway, the
approval gate, the RAG citation gate) — this package supplies the identity,
the policy matrix, and the visibility (audit rows, injection findings) those
boundaries check against and log to.
"""

from app.security.audit import AuditLog, AuditRow, build_default_audit_log
from app.security.auth import AuthError, TokenAuthenticator, build_default_authenticator
from app.security.injection import InjectionFinding, screen_chunks, screen_untrusted_content, screen_user_input
from app.security.permissions import check, has_permission, permission_matrix, permissions_for

__all__ = [
    "AuditLog",
    "AuditRow",
    "AuthError",
    "InjectionFinding",
    "TokenAuthenticator",
    "build_default_audit_log",
    "build_default_authenticator",
    "check",
    "has_permission",
    "permission_matrix",
    "permissions_for",
    "screen_chunks",
    "screen_untrusted_content",
    "screen_user_input",
]
