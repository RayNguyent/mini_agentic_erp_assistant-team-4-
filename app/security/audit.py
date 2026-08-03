"""Append-only audit log.

One JSONL row per security-relevant event: auth, tool calls, approval
decisions, blocked actions, final responses. Append-only and separate from the
trace store (app.observability.trace) — traces are for debugging a request's
behavior, the audit log is for answering "who did what, when" under review,
and the two must not be conflatable into one file that gets rotated or
truncated for performance reasons.

Critical ordering rule: an approval decision is written here *before* the
pending-approval record is deleted from ApprovalStore (app/approvals/store.py
pop() is destructive) — otherwise a resolved approval leaves no trace at all
if the audit write were to happen after the pop and something failed in
between.
"""

import json
import threading
import time
from pathlib import Path

from pydantic import BaseModel

AUDIT_LOG_PATH = Path("data/audit.jsonl")


class AuditRow(BaseModel):
    timestamp: float
    category: str  # "auth" | "tool_call" | "approval" | "blocked" | "response" | "memory_write"
    actor_id: str | None = None
    actor_role: str | None = None
    request_id: str | None = None
    action: str
    outcome: str  # "success" | "denied" | "error" | "pending" | "approved" | "rejected"
    detail: dict = {}


class AuditLog:
    """Thread-safe append-only writer. One lock, one file — the same
    single-process assumption app/approvals/store.py already documents."""

    def __init__(self, path: Path | str = AUDIT_LOG_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._rows: list[AuditRow] = []  # in-memory mirror, for /audit and tests

    def record(
        self,
        category: str,
        action: str,
        outcome: str,
        *,
        actor=None,
        request_id: str | None = None,
        detail: dict | None = None,
    ) -> AuditRow:
        row = AuditRow(
            timestamp=time.time(),
            category=category,
            actor_id=actor.user_id if actor else None,
            actor_role=actor.role if actor else None,
            request_id=request_id,
            action=action,
            outcome=outcome,
            detail=detail or {},
        )
        with self._lock:
            self._rows.append(row)
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(row.model_dump_json() + "\n")
        return row

    def recent(self, limit: int = 100, category: str | None = None) -> list[AuditRow]:
        rows = self._rows if category is None else [r for r in self._rows if r.category == category]
        return rows[-limit:]

    # --- convenience wrappers, one per audit category the spec names -------

    def auth(self, actor, outcome: str, *, request_id: str | None = None) -> AuditRow:
        return self.record("auth", "authenticate", outcome, actor=actor, request_id=request_id)

    def tool_call(self, tool_name: str, outcome: str, *, actor=None, request_id=None, detail=None) -> AuditRow:
        return self.record("tool_call", tool_name, outcome, actor=actor, request_id=request_id, detail=detail)

    def approval(self, approval_id: str, outcome: str, *, actor=None, request_id=None, detail=None) -> AuditRow:
        return self.record(
            "approval", "create_risk", outcome, actor=actor, request_id=request_id,
            detail={"approval_id": approval_id, **(detail or {})},
        )

    def blocked(self, reason: str, *, actor=None, request_id=None, detail=None) -> AuditRow:
        return self.record("blocked", reason, "denied", actor=actor, request_id=request_id, detail=detail)

    def response(self, route: str, *, actor=None, request_id=None, detail=None) -> AuditRow:
        return self.record("response", route, "success", actor=actor, request_id=request_id, detail=detail)

    def memory_write(self, decision, *, actor=None, request_id=None) -> AuditRow:
        """`outcome` here is the MemoryDecision's action value (save/update/
        ignore/expire/reject) — a source-specific vocabulary, same as
        `approval`'s pending/approved/rejected differs from `tool_call`'s
        success/error. A REJECT (poisoning attempt) is recorded exactly like
        any other decision, not treated as an error outcome."""
        return self.record(
            "memory_write", decision.candidate.source, decision.action.value,
            actor=actor, request_id=request_id,
            detail={
                "reason": decision.reason,
                "subject": decision.candidate.subject,
                "target_memory_id": decision.target_memory_id,
            },
        )


def build_default_audit_log() -> AuditLog:
    return AuditLog()
