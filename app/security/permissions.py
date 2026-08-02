"""Role -> permission matrix, and the check every tool call and RAG query
goes through.

One matrix, read by three places: app.agents.base.authorize (specialist
dispatch), app.tools (a defence-in-depth check at the actual tool boundary —
see check_tool_permission), and app.rag.acl (document classification, a
parallel but distinct scope). Keeping the source of one place is what makes
"add a new permission" a one-line change instead of a hunt across modules.
"""

from app.state import Role

# Baseline roles per final-project.pdf §7: "A developer, project manager, and
# auditor role is enough for the baseline."
PERMISSIONS: dict[Role, frozenset[str]] = {
    "developer": frozenset({
        "project.read",
        "project.risk.read",
        "document.read",
    }),
    "project_manager": frozenset({
        "project.read",
        "project.risk.read",
        "project.risk.create",
        "project.finance.read",
        "document.read",
    }),
    "auditor": frozenset({
        "project.read",
        "project.risk.read",
        "project.finance.read",
        "document.read",
        # Auditors can see everything relevant to oversight, but cannot
        # trigger a write — auditing and acting are kept separate.
    }),
}


def permissions_for(role: Role | str | None) -> frozenset[str]:
    if role is None:
        return frozenset()
    return PERMISSIONS.get(role, frozenset())


def has_permission(role: Role | str | None, permission: str) -> bool:
    return permission in permissions_for(role)


def check(actor, permission: str) -> bool:
    """The function app.graph.engine.GraphContext(authorize=...) is wired
    with. `actor` may be None for an unauthenticated caller, which holds no
    permissions — least privilege, matching app.rag.acl's default."""
    if actor is None:
        return False
    return has_permission(actor.role, permission)


def permission_matrix() -> dict[str, list[str]]:
    """Rendered into docs/threat-model.md's permission matrix table."""
    return {role: sorted(perms) for role, perms in PERMISSIONS.items()}
