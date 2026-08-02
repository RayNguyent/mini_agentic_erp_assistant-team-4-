"""Document access control, applied before generation.

The one rule that matters here: a chunk the caller may not see never enters the
prompt. Filtering the *answer* instead of the *context* means the model was
shown restricted content and merely asked not to repeat it — which is a request,
not a control. Every retrieval path in app/rag/retrieve.py filters at recall.
"""

from app.rag.documents import CLASSIFICATIONS, Chunk
from app.state import Role

# Role -> the classifications that role may read.
#
# developer:       delivery material only.
# project_manager: adds finance, needed to answer budget questions.
# auditor:         adds restricted; the restricted material names audit as an
#                  authorised recipient, and an auditor who cannot see the
#                  evidence cannot audit.
_CLEARANCE: dict[str, frozenset[str]] = {
    "developer": frozenset({"public", "internal"}),
    "project_manager": frozenset({"public", "internal", "finance"}),
    "auditor": frozenset({"public", "internal", "finance", "restricted"}),
}

# Unknown or absent role: least privilege, not most.
_DEFAULT_CLEARANCE = frozenset({"public"})


def clearance_for(role: Role | str | None) -> frozenset[str]:
    if role is None:
        return _DEFAULT_CLEARANCE
    return _CLEARANCE.get(str(role), _DEFAULT_CLEARANCE)


def can_read(role: Role | str | None, classification: str) -> bool:
    return classification in clearance_for(role)


def filter_chunks(
    chunks: list[Chunk], role: Role | str | None
) -> tuple[list[Chunk], list[str]]:
    """Return `(visible_chunks, denied_chunk_ids)`.

    The denied ids are returned rather than discarded so the trace can record
    that filtering happened and how much it removed — "zero restricted results"
    and "the filter never ran" are otherwise indistinguishable in evidence.
    """
    allowed = clearance_for(role)
    visible: list[Chunk] = []
    denied: list[str] = []
    for chunk in chunks:
        if chunk.classification in allowed:
            visible.append(chunk)
        else:
            denied.append(chunk.chunk_id)
    return visible, denied


def clearance_matrix() -> dict[str, list[str]]:
    """Rendered into the permission matrix in docs/threat-model.md."""
    return {
        role: [c for c in CLASSIFICATIONS if c in allowed]
        for role, allowed in _CLEARANCE.items()
    }
