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
from app.memory.policy import MemoryCandidate
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
                # First pass, pre-approval: no memory write here. The real
                # approval_id doesn't exist yet at this point — ApprovalStore
                # .create() is only called later, in app.api.routes.chat(),
                # once the whole graph suspends — so there is nothing to set
                # pending_approval_id to yet.
                return AgentResult(agent=self.name, ok=True, awaiting_approval=True)

            # Resume pass (state.approved is not None): the approval decision
            # has already been made, so any working-memory placeholder for it
            # is stale regardless of outcome.
            self._forget_pending_approval(ctx, state)

            if result_state.error_code is not None:
                return AgentResult(
                    agent=self.name,
                    ok=result_state.error_code == ErrorCode.APPROVAL_REJECTED,
                    tool_used="create_risk",
                    error_code=result_state.error_code,
                    error_message=result_state.error_message,
                )

            self._remember(ctx, state, tool_input, result_state.tool_output)

            return AgentResult(
                agent=self.name,
                ok=True,
                tool_used="create_risk",
                tool_output=result_state.tool_output,
            )

        return timed(self.name, _run)

    def _forget_pending_approval(self, ctx: GraphContext, state: AgentState) -> None:
        memory_store = ctx.get("memory_store")
        if memory_store is not None and state.actor is not None:
            memory_store.clear_working(state.actor.user_id, "pending_approval_id")

    def _remember(self, ctx: GraphContext, state: AgentState, tool_input: dict | None, tool_output: dict | None) -> None:
        memory_store = ctx.get("memory_store")
        if memory_store is None or state.actor is None:
            return
        tool_input = tool_input or {}
        project_code = tool_input.get("project_code")
        payload = tool_input.get("risk_payload") or {}
        candidate = MemoryCandidate(
            text=f"Risk created for {project_code}: {payload.get('title', '')} (severity {payload.get('severity', 'unknown')})",
            source="tool_result",
            confidence=0.95,
            subject=project_code or "risk",
        )
        decision = memory_store.propose(state.actor.user_id, candidate)
        if ctx.trace is not None:
            ctx.trace.record_memory_decision(decision, agent=self.name)
        audit = ctx.get("audit_log")
        if audit is not None:
            audit.memory_write(decision, actor=state.actor, request_id=state.request_id)

    @property
    def awaiting_approval_marker(self) -> str:
        # Exposed for the supervisor to recognise a pending write without
        # importing NextAction directly.
        return NextAction.AWAIT_APPROVAL.value
