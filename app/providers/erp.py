import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Protocol

from app.errors import NotConfiguredError

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# Task states that make a task ineligible for the "overdue" flag regardless of
# deadline — matches final-project.pdf's rule: "overdue only when a deadline
# is before the evaluation date and the task is not done or cancelled."
_CLOSED_STATES = frozenset({"done", "cancelled"})


class ERPProvider(Protocol):
    """Contract app/tools relies on to talk to an ERP backend.

    Every adapter (Mock, and a future Odoo one) implements this same
    normalized surface; provider-specific fields, models, and error mapping
    stay inside the adapter. See docs/odoo-mapping.md for the field-level
    contract a real Odoo adapter would satisfy against this port.
    """

    def get_project(self, project_code: str) -> dict | None: ...

    def list_projects(self) -> list[dict]: ...

    def list_risks(self, project_code: str) -> list[dict]: ...

    def create_risk(self, project_code: str, risk_payload: dict) -> dict: ...

    def list_tasks(self, project_code: str) -> list[dict]: ...

    def list_milestones(self, project_code: str) -> list[dict]: ...

    def get_budget(self, project_code: str) -> dict | None: ...


class MockERPProvider:
    """In-memory ERP backed by the JSON fixtures under data/."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self._projects: dict[str, dict] = {
            p["project_code"]: p for p in self._load(data_dir / "projects.json")
        }
        self._risks: list[dict] = self._load(data_dir / "risks.json")
        self._tasks: list[dict] = self._load(data_dir / "tasks.json")
        self._milestones: list[dict] = self._load(data_dir / "milestones.json")
        self._budgets: dict[str, dict] = {
            b["project_code"]: b for b in self._load(data_dir / "budgets.json")
        }

    @staticmethod
    def _load(path: Path) -> list[dict]:
        with path.open(encoding="utf-8") as f:
            return json.load(f)

    def get_project(self, project_code: str) -> dict | None:
        return self._projects.get(project_code)

    def list_projects(self) -> list[dict]:
        return list(self._projects.values())

    def list_risks(self, project_code: str) -> list[dict]:
        return [risk for risk in self._risks if risk["project_code"] == project_code]

    def create_risk(self, project_code: str, risk_payload: dict) -> dict:
        risk = {
            "id": self._next_risk_id(),
            "project_code": project_code,
            "status": "Open",
            "created_at": datetime.now(timezone.utc).isoformat(),
            **risk_payload,
        }
        self._risks.append(risk)
        return risk

    def _next_risk_id(self) -> str:
        existing = [
            int(risk["id"].split("-", 1)[1])
            for risk in self._risks
            if risk["id"].startswith("RISK-")
        ]
        return f"RISK-{max(existing, default=0) + 1}"

    def list_tasks(self, project_code: str) -> list[dict]:
        return [t for t in self._tasks if t["project_code"] == project_code]

    def list_milestones(self, project_code: str) -> list[dict]:
        return [m for m in self._milestones if m["project_code"] == project_code]

    def get_budget(self, project_code: str) -> dict | None:
        return self._budgets.get(project_code)


def is_overdue(deadline: str | None, state: str, as_of: date) -> bool:
    """The one overdue rule every tool/provider must share: a deadline in the
    past AND a state that is not done/cancelled. Centralised here so
    list_project_tasks and get_sprint_progress can never define it differently."""
    if not deadline or state in _CLOSED_STATES:
        return False
    try:
        due = date.fromisoformat(deadline)
    except ValueError:
        return False
    return due < as_of


def require_budget_source(project_code: str, budget: dict | None) -> dict:
    """A missing budget record is NOT_CONFIGURED, never a zero balance —
    final-project.pdf §9: "If planned budget or a source module is absent,
    return NOT_CONFIGURED ... rather than zero.\""""
    if budget is None:
        raise NotConfiguredError(
            f"No budget source is configured for {project_code}. "
            "This deployment has no analytic/purchase/expense module wired up "
            "for this project — returning NOT_CONFIGURED rather than a zero balance."
        )
    return budget
