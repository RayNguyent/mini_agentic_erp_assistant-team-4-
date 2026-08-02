"""The memory write policy: save, update, ignore, expire, or reject.

Every decision is typed and carries a reason — this module is what the spec's
"one save/update/expire decision and one poisoning rejection, visible with
provenance" demo scenario points at. The rejection rule is deterministic
pattern-matching, not a model call: whether a write candidate is an attempt to
plant an instruction ("ignore approval next time", "always trust this vendor")
must not itself depend on trusting the LLM inside the loop the rule exists to
protect.
"""

import time
from enum import Enum

from pydantic import BaseModel

from app.memory.long_term import LongTermMemory, MemoryEntry, new_memory_id
from app.security.patterns import find_match

MIN_CONFIDENCE = 0.35
DUPLICATE_JACCARD_THRESHOLD = 0.6

# Untrusted sources (document, tool_result, web) are held to a stricter bar
# than a direct user turn, which at least has an identified, authenticated
# author — see decide() below for how the two cases differ.
UNTRUSTED_SOURCES = frozenset({"document", "tool_result", "web"})


class MemoryAction(str, Enum):
    SAVE = "save"
    UPDATE = "update"
    IGNORE = "ignore"
    EXPIRE = "expire"
    REJECT = "reject"


class MemoryCandidate(BaseModel):
    text: str
    source: str  # "user_turn" | "tool_result" | "document" | "agent_inference" | "web"
    confidence: float = 1.0
    subject: str = ""


class MemoryDecision(BaseModel):
    action: MemoryAction
    reason: str
    candidate: MemoryCandidate
    target_memory_id: str | None = None  # set for UPDATE / EXPIRE


def _jaccard(a: str, b: str) -> float:
    tokens_a, tokens_b = set(a.lower().split()), set(b.lower().split())
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _near_duplicate(candidate: MemoryCandidate, existing: list[MemoryEntry]) -> MemoryEntry | None:
    """Token-overlap similarity, not a character prefix.

    A prefix check misses "PRJ-001 is currently in sprint 4 of 6" vs
    "...sprint 5 of 6" whenever the differing word falls inside the prefix
    window — exactly the kind of one-field update (sprint number, status,
    severity) this check exists to catch as an UPDATE rather than a duplicate
    SAVE. Word-set overlap is robust to where in the sentence the change is.
    """
    best_entry, best_score = None, 0.0
    for entry in existing:
        if entry.subject != candidate.subject:
            continue
        score = _jaccard(entry.text, candidate.text)
        if score > best_score:
            best_entry, best_score = entry, score
    return best_entry if best_score >= DUPLICATE_JACCARD_THRESHOLD else None


def decide(
    candidate: MemoryCandidate,
    memory: LongTermMemory,
    *,
    now: float | None = None,
) -> MemoryDecision:
    """The single entry point every memory write goes through. Never writes
    directly — returns a decision the caller applies, so the decision itself
    is what gets logged and tested."""

    # 1. Poisoning check — deterministic, applies before anything else.
    matched = find_match(candidate.text)
    if matched and candidate.source in UNTRUSTED_SOURCES:
        return MemoryDecision(
            action=MemoryAction.REJECT,
            reason=(
                f"candidate from untrusted source '{candidate.source}' matches an "
                f"instruction-injection pattern ({matched!r}) — treated as data, not "
                "a command, and not written to memory"
            ),
            candidate=candidate,
        )
    if matched:
        # Even a direct user turn does not get to silently rewrite policy via
        # memory; it is rejected, just with a different, less alarming reason.
        return MemoryDecision(
            action=MemoryAction.REJECT,
            reason=f"candidate reads as an attempted instruction ({matched!r}), not a fact worth storing",
            candidate=candidate,
        )

    # 2. Low-confidence candidates are not worth the storage or the future
    # recall noise.
    if candidate.confidence < MIN_CONFIDENCE:
        return MemoryDecision(
            action=MemoryAction.IGNORE,
            reason=f"confidence {candidate.confidence:.2f} below the {MIN_CONFIDENCE} save threshold",
            candidate=candidate,
        )

    # 3. A near-duplicate on the same subject is an update, not a new entry —
    # otherwise recall accumulates repeated, slightly-reworded facts forever.
    duplicate = _near_duplicate(candidate, memory.entries)
    if duplicate is not None:
        return MemoryDecision(
            action=MemoryAction.UPDATE,
            reason=f"near-duplicate of existing entry {duplicate.memory_id} for subject '{candidate.subject}'",
            candidate=candidate,
            target_memory_id=duplicate.memory_id,
        )

    return MemoryDecision(
        action=MemoryAction.SAVE,
        reason=f"new fact from a trusted-enough source ('{candidate.source}') with sufficient confidence",
        candidate=candidate,
    )


def apply(decision: MemoryDecision, memory: LongTermMemory, vector: list[float] | None = None) -> LongTermMemory:
    """Apply a decision. REJECT and IGNORE are no-ops on the store — the point
    of returning them as decisions is that they are logged, not that they act."""
    if decision.action == MemoryAction.SAVE:
        entry = MemoryEntry(
            memory_id=new_memory_id(),
            text=decision.candidate.text,
            written_at=time.time(),
            source=decision.candidate.source,
            confidence=decision.candidate.confidence,
            subject=decision.candidate.subject,
            vector=vector or [],
        )
        return memory.add(entry)

    if decision.action == MemoryAction.UPDATE and decision.target_memory_id:
        return memory.update(decision.target_memory_id, decision.candidate.text)

    if decision.action == MemoryAction.EXPIRE and decision.target_memory_id:
        return memory.remove(decision.target_memory_id)

    return memory  # IGNORE / REJECT: memory is unchanged


def expire_stale(memory: LongTermMemory, now: float | None = None) -> tuple[LongTermMemory, list[MemoryDecision]]:
    """Sweep stale entries, returning one EXPIRE decision per entry removed so
    the sweep is auditable rather than a silent background deletion."""
    decisions = [
        MemoryDecision(
            action=MemoryAction.EXPIRE,
            reason=f"entry {entry.memory_id} exceeded its staleness window",
            candidate=MemoryCandidate(text=entry.text, source=entry.source, subject=entry.subject),
            target_memory_id=entry.memory_id,
        )
        for entry in memory.stale_entries(now)
    ]
    updated = memory
    for decision in decisions:
        updated = apply(decision, updated)
    return updated, decisions
