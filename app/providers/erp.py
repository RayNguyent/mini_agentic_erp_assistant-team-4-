import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


class ERPProvider(Protocol):
    """Contract app/tools relies on to talk to an ERP backend."""

    def get_project(self, project_code: str) -> dict | None: ...

    def list_projects(self) -> list[dict]: ...

    def list_risks(self, project_code: str) -> list[dict]: ...

    def create_risk(self, project_code: str, risk_payload: dict) -> dict: ...


class MockERPProvider:
    """In-memory ERP backed by the JSON fixtures under data/."""

    def __init__(self, data_dir: Path = DATA_DIR):
        self._projects: dict[str, dict] = {
            p["project_code"]: p for p in self._load(data_dir / "projects.json")
        }
        self._risks: list[dict] = self._load(data_dir / "risks.json")

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
