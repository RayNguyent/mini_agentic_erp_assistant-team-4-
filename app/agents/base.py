"""The specialist-agent contract.

Each specialist declares what it is allowed to touch — a tool allowlist and a
required permission — rather than being handed the whole registry. That single
decision does most of the security work in the multi-agent layer: the document
researcher physically cannot call `create_risk`, because the registry it
receives does not contain it. Confinement is structural, not a prompt asking
the model to stay in its lane.
"""

import time
from typing import Protocol, runtime_checkable

from app.errors import ErrorCode
from app.graph.engine import GraphContext
from app.state import AgentResult, AgentState, AgentStep


@runtime_checkable
class SpecialistAgent(Protocol):
    """One bounded capability the supervisor can dispatch to."""

    name: str
    description: str
    allowed_tools: frozenset[str]
    required_permission: str

    def run(self, step: AgentStep, state: AgentState, ctx: GraphContext) -> AgentResult: ...


class ScopedRegistry:
    """A tool registry narrowed to one agent's allowlist.

    Wrapping rather than filtering keeps the underlying registry authoritative:
    a tool added later is unreachable to an agent whose allowlist was not also
    updated, which is the safe direction for that mistake to fail in.
    """

    def __init__(self, registry, allowed: frozenset[str]) -> None:
        self._registry = registry
        self._allowed = allowed

    def get(self, tool_name: str):
        if tool_name not in self._allowed:
            return None
        return self._registry.get(tool_name)

    @property
    def allowed(self) -> frozenset[str]:
        return self._allowed


# What each permission actually lets someone do, in plain language — shown to
# the user instead of the raw permission string.
_PERMISSION_LABELS: dict[str, str] = {
    "project.read": "view project details",
    "project.risk.read": "view project risks",
    "project.risk.create": "create or edit project risks",
    "project.finance.read": "view budget and financial data",
    "document.read": "read project documents",
}


def _humanize_role(role: str) -> str:
    return role.replace("_", " ").title()


def denied(agent: str, permission: str, actor) -> AgentResult:
    """A refusal that explains, in plain language, what the caller can't do
    and who could do it instead — not just the raw permission string."""
    from app.security.permissions import PERMISSIONS  # local import: keeps this module decoupled from app.security at load time

    label = _PERMISSION_LABELS.get(permission, f"do this (requires '{permission}')")
    who = f"Your role ('{_humanize_role(actor.role)}')" if actor is not None else "You"
    message = f"{who} can't {label}."

    holder_roles = sorted(role for role, perms in PERMISSIONS.items() if permission in perms)
    if holder_roles:
        readable = (
            _humanize_role(holder_roles[0])
            if len(holder_roles) == 1
            else ", ".join(_humanize_role(r) for r in holder_roles[:-1]) + f" or {_humanize_role(holder_roles[-1])}"
        )
        message += f" Ask someone with the {readable} role, or sign in with a different role, for this."

    return AgentResult(
        agent=agent,
        ok=False,
        error_code=ErrorCode.FORBIDDEN,
        error_message=message,
    )


def authorize(agent: SpecialistAgent, state: AgentState, ctx: GraphContext) -> bool:
    """Check the caller against the agent's required permission.

    The check lives here, before dispatch, so an unauthorised request never
    reaches the tool boundary at all. `ctx` supplies the policy function so this
    module does not depend on the security package.
    """
    return authorize_permission(state.actor, agent.required_permission, ctx)


def authorize_permission(actor, permission: str, ctx: GraphContext) -> bool:
    """Check one explicit permission string, not an agent's default.

    ERPAnalyst needs this: it wraps five tools that do not all require the
    same permission (four need `project.read`, `get_budget_summary` needs the
    stricter `project.finance.read`), so the class-level `required_permission`
    alone is too coarse — a developer denied budget data would otherwise pass
    the agent-level check and only get denied deeper, or not at all.
    """
    check = ctx.get("authorize")
    if check is None:
        return True
    return bool(check(actor, permission))


def timed(agent_name: str, fn) -> AgentResult:
    """Run an agent body, recording elapsed time and converting an unexpected
    exception into a failed AgentResult rather than collapsing the whole graph.

    One specialist failing should degrade the answer, not the request: the
    synthesizer can still report what the other specialists found.
    """
    started = time.perf_counter()
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001 - contained on purpose
        result = AgentResult(
            agent=agent_name,
            ok=False,
            error_code=ErrorCode.UNKNOWN,
            error_message=f"{type(exc).__name__}: {exc}",
        )
    return result.model_copy(
        update={"elapsed_ms": round((time.perf_counter() - started) * 1000, 1)}
    )
