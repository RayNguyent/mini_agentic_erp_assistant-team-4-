import threading
import uuid
from datetime import datetime, timezone

from app.approvals.models import PendingApproval
from app.state import AgentState


class ApprovalNotFoundError(Exception):
    """No pending approval for this id — either it never existed or it was
    already approved/rejected."""


class ApprovalStore:
    """In-memory store for states halted at NextAction.AWAIT_APPROVAL.

    Single-process only: the dict lives in this process's memory, so the API
    must run with a single uvicorn worker.
    """

    def __init__(self):
        self._pending: dict[str, PendingApproval] = {}
        self._lock = threading.Lock()

    def create(self, state: AgentState) -> str:
        approval_id = str(uuid.uuid4())
        with self._lock:
            self._pending[approval_id] = PendingApproval(
                id=approval_id,
                state=state,
                created_at=datetime.now(timezone.utc),
            )
        return approval_id

    def pop(self, approval_id: str) -> AgentState:
        """Get-and-delete, so an id can only be resolved once."""
        with self._lock:
            pending = self._pending.pop(approval_id, None)
        if pending is None:
            raise ApprovalNotFoundError(f"No pending approval with id '{approval_id}'.")
        return pending.state


def build_default_store() -> ApprovalStore:
    return ApprovalStore()
