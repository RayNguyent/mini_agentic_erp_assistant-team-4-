"""The RiskWriter specialist: the sole path to create_risk.

Deliberately does not introduce a new approval mechanism — it drives
app.runtime.execute_write_tool, the same function the single-agent graph uses,
so create_risk has exactly one approval code path regardless of which route
got a request there. A second implementation of "hold pending until approved"
is a second place that gate could be gotten wrong.
"""

from app.agents.base import ScopedRegistry, authorize, denied, timed
from app.errors import ErrorCode
from app.graph.engine import GraphContext
from app.runtime import WRITE_TOOLS, execute_write_tool
from app.state import AgentResult, AgentState, AgentStep, NextAction

WRITE_TOOL_NAMES = frozenset(WRITE_TOOLS.values())


class RiskWriter:
    name = "risk_writer"
    description = "Creates a risk record. Always requires human approval before executing."
    allowed_tools: frozenset[str] = WRITE_TOOL_NAMES
    required_permission = "project.risk.create"

    def run(self, step: AgentStep, state: AgentState, ctx: GraphContext) -> AgentResult:
        if not authorize(self, state, ctx):
            return denied(self.name, self.required_permission, state.actor)

        def _run() -> AgentResult:
            registry = ScopedRegistry(ctx.require("tool_registry"), self.allowed_tools)
            # `state.tool_input` is what the API merges the human-supplied
            # risk_payload into on /approve (mirroring the single-agent
            # resume() flow, app/api/routes.py::_approval_response). The
            # plan step's own `inputs.tool_input` is only the pre-approval
            # draft and must not be used once a decision has been made.
            tool_input = state.tool_input if state.approved is not None else (
                step.inputs.get("tool_input") or {}
            )
            probe = AgentState(
                intent="create_risk",
                selected_tool="create_risk",
                tool_input=tool_input,
                approval_required=True,
                approved=state.approved,
                next_action=NextAction.EXECUTE_WRITE_TOOL,
                risk_level="high",
            )
            result_state = execute_write_tool(probe, registry, ctx=ctx)

            if result_state.next_action == NextAction.AWAIT_APPROVAL:
                return AgentResult(agent=self.name, ok=True, awaiting_approval=True)

            if result_state.error_code is not None:
                return AgentResult(
                    agent=self.name,
                    ok=result_state.error_code == ErrorCode.APPROVAL_REJECTED,
                    tool_used="create_risk",
                    error_code=result_state.error_code,
                    error_message=result_state.error_message,
                )

            return AgentResult(
                agent=self.name,
                ok=True,
                tool_used="create_risk",
                tool_output=result_state.tool_output,
            )

        return timed(self.name, _run)

    @property
    def awaiting_approval_marker(self) -> str:
        # Exposed for the supervisor to recognise a pending write without
        # importing NextAction directly.
        return NextAction.AWAIT_APPROVAL.value
