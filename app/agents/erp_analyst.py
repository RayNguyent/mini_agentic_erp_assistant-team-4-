"""The ERPAnalyst specialist: read-only ERP/project tools.

Reuses app.runtime's tool execution and retry path rather than reimplementing
it — the same bounded-retry-with-backoff logic that the single-agent graph
uses. The only new thing here is the allowlist: ERPAnalyst is wired with a
ScopedRegistry containing only the five read tools, so it cannot reach
create_risk even if a prompt injection convinced the underlying model to try.
"""

from app.agents.base import ScopedRegistry, authorize_permission, denied, timed
from app.graph.engine import GraphContext
from app.graph.retry import RetryPolicy
from app.runtime import DEFAULT_RETRY_POLICY, READ_TOOLS, _run_tool_with_retry
from app.state import AgentResult, AgentState, AgentStep, NextAction
from app.tools.specs import permission_for

READ_TOOL_NAMES = frozenset(READ_TOOLS.values())

# A step's `inputs.intent` selects which read tool to call, matching the same
# intent vocabulary the single-agent classifier produces.
_INTENT_TO_TOOL = dict(READ_TOOLS)


class ERPAnalyst:
    name = "erp_analyst"
    description = "Looks up project status, tasks, sprint progress, budget, and risks."
    allowed_tools: frozenset[str] = READ_TOOL_NAMES
    required_permission = "project.read"

    def run(self, step: AgentStep, state: AgentState, ctx: GraphContext) -> AgentResult:
        # Permission is checked per resolved tool, not once at the class
        # level: this agent wraps five tools that do not all require the same
        # permission (get_budget_summary needs project.finance.read).
        tool_name = self._resolve_tool(step)
        if tool_name is None:
            return AgentResult(
                agent=self.name, ok=False,
                error_message="Could not determine which project tool to call.",
            )

        permission = permission_for(tool_name) or self.required_permission
        if not authorize_permission(state.actor, permission, ctx):
            return denied(self.name, permission, state.actor)

        def _run() -> AgentResult:
            registry = ScopedRegistry(ctx.require("tool_registry"), self.allowed_tools)
            probe = AgentState(
                intent="probe",
                selected_tool=tool_name,
                tool_input=step.inputs.get("tool_input") or {},
                next_action=NextAction.EXECUTE_READ_TOOL,
            )
            # `or`, not `ctx.get(key, default)`: a caller that explicitly
            # passes retry_policy=None (e.g. an unset kwarg forwarded as-is)
            # would otherwise store a real `None` in the context, which
            # `.get(key, default)` treats as present and returns unchanged —
            # `default` only fires when the key is *absent*, not when its
            # value is falsy. This broke retry entirely until it was caught
            # by eval case retry-07 (see eval/golden/cases.v1.json).
            policy: RetryPolicy = ctx.get("retry_policy") or DEFAULT_RETRY_POLICY
            result_state = _run_tool_with_retry(probe, registry, policy=policy, ctx=ctx)

            if result_state.error_code is not None:
                return AgentResult(
                    agent=self.name,
                    ok=False,
                    tool_used=tool_name,
                    error_code=result_state.error_code,
                    error_message=result_state.error_message,
                    retries=result_state.retry_count,
                )

            return AgentResult(
                agent=self.name,
                ok=True,
                tool_used=tool_name,
                tool_output=result_state.tool_output,
                retries=result_state.retry_count,
            )

        return timed(self.name, _run)

    def _resolve_tool(self, step: AgentStep) -> str | None:
        explicit = step.inputs.get("tool")
        if explicit in self.allowed_tools:
            return explicit
        intent = step.inputs.get("intent")
        if intent in _INTENT_TO_TOOL:
            return _INTENT_TO_TOOL[intent]
        return None
