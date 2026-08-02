"""Short-term memory: the conversation buffer.

Not "the last N turns" — allow-list compaction. When a conversation exceeds the
turn budget, older turns are summarised, but a fixed set of fields never gets
compacted away: a pending approval, a safety/refusal flag, or an active project
code. A generic "summarize everything" pass is how an approval-in-flight
silently disappears from context — the model would have no way to know one was
ever pending.
"""

from pydantic import BaseModel, Field

MAX_TURNS = 8

# Turn attributes that survive compaction verbatim rather than being folded
# into the prose summary. Anything not on this list is compactable.
ALWAYS_KEPT_FLAGS = frozenset({"pending_approval", "refused", "risk_level_high"})


class Turn(BaseModel):
    role: str
    content: str
    flags: frozenset[str] = Field(default_factory=frozenset)

    model_config = {"frozen": True}


class ConversationBuffer(BaseModel):
    turns: list[Turn] = Field(default_factory=list)
    max_turns: int = MAX_TURNS
    summary: str = ""

    def append(self, role: str, content: str, flags: frozenset[str] = frozenset()) -> "ConversationBuffer":
        return self.model_copy(update={"turns": [*self.turns, Turn(role=role, content=content, flags=flags)]})

    @property
    def flagged_turns(self) -> list[Turn]:
        return [t for t in self.turns if t.flags & ALWAYS_KEPT_FLAGS]

    def compact(self, summarizer=None) -> "ConversationBuffer":
        """Fold everything beyond `max_turns` into `summary`, except flagged
        turns, which are kept verbatim regardless of age."""
        if len(self.turns) <= self.max_turns:
            return self

        cutoff = len(self.turns) - self.max_turns
        to_compact = [t for t in self.turns[:cutoff] if not (t.flags & ALWAYS_KEPT_FLAGS)]
        kept_old = [t for t in self.turns[:cutoff] if t.flags & ALWAYS_KEPT_FLAGS]
        recent = self.turns[cutoff:]

        if not to_compact:
            return self

        new_summary_text = (
            summarizer(to_compact) if summarizer else _extractive_summary(to_compact)
        )
        combined_summary = f"{self.summary}\n{new_summary_text}".strip() if self.summary else new_summary_text

        return self.model_copy(update={"turns": [*kept_old, *recent], "summary": combined_summary})

    def as_messages(self) -> list[dict]:
        messages = []
        if self.summary:
            messages.append({"role": "system", "content": f"Earlier conversation summary: {self.summary}"})
        messages.extend({"role": t.role, "content": t.content} for t in self.turns)
        return messages


def _extractive_summary(turns: list[Turn]) -> str:
    user_lines = [t.content for t in turns if t.role == "user"]
    if not user_lines:
        return "Prior turns discussed no user requests worth summarising."
    return "User previously asked about: " + "; ".join(line[:80] for line in user_lines[-3:])
