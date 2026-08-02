from app.providers.erp import ERPProvider, MockERPProvider
from app.tools.erp_tools import (
    make_create_risk,
    make_get_budget_summary,
    make_get_project_status,
    make_get_sprint_progress,
    make_list_project_tasks,
    make_list_risks,
)


class ToolRegistry:
    """Concrete ToolRegistry satisfying the Protocol app.runtime depends on."""

    def __init__(self, provider: ERPProvider):
        self._tools = {
            "get_project_status": make_get_project_status(provider),
            "list_risks": make_list_risks(provider),
            "create_risk": make_create_risk(provider),
            "list_project_tasks": make_list_project_tasks(provider),
            "get_sprint_progress": make_get_sprint_progress(provider),
            "get_budget_summary": make_get_budget_summary(provider),
        }

    def get(self, tool_name: str):
        return self._tools.get(tool_name)


def build_default_registry() -> ToolRegistry:
    return ToolRegistry(MockERPProvider())
