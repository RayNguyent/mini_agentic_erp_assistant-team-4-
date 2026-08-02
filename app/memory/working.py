"""Working memory: per-session task/project facts, TTL-expiring.

The scratchpad of "what are we in the middle of" — the active project code, an
approval waiting on a decision, the last tool result — as opposed to durable
facts about the world (long-term) or the raw conversation (short-term). Every
entry has an expiry; nothing here is meant to outlive the session it was
written in.
"""

import time

from pydantic import BaseModel, Field

DEFAULT_TTL_S = 30 * 60  # 30 minutes: long enough for one working session


class WorkingItem(BaseModel):
    key: str
    value: object
    written_at: float
    ttl_s: float = DEFAULT_TTL_S
    source: str = "assistant"  # what produced it: user turn, tool call, agent

    def expired(self, now: float | None = None) -> bool:
        return (now or time.time()) - self.written_at > self.ttl_s


class WorkingMemory(BaseModel):
    """A small typed key-value store, not a general cache: keys are the fixed
    set of session facts the runtime actually reads (active_project_code,
    pending_approval_id, last_tool, last_tool_output)."""

    items: dict[str, WorkingItem] = Field(default_factory=dict)

    def set(self, key: str, value: object, *, source: str = "assistant", ttl_s: float = DEFAULT_TTL_S) -> "WorkingMemory":
        item = WorkingItem(key=key, value=value, written_at=time.time(), ttl_s=ttl_s, source=source)
        return self.model_copy(update={"items": {**self.items, key: item}})

    def get(self, key: str, now: float | None = None) -> object | None:
        item = self.items.get(key)
        if item is None or item.expired(now):
            return None
        return item.value

    def expire_stale(self, now: float | None = None) -> tuple["WorkingMemory", list[str]]:
        """Drop expired entries. Returns the new memory and the keys removed,
        so a caller can log the expire decision rather than have it happen
        silently."""
        now = now or time.time()
        live = {k: v for k, v in self.items.items() if not v.expired(now)}
        removed = [k for k in self.items if k not in live]
        return self.model_copy(update={"items": live}), removed

    def clear(self, key: str) -> "WorkingMemory":
        if key not in self.items:
            return self
        return self.model_copy(update={"items": {k: v for k, v in self.items.items() if k != key}})
