"""Per-user memory: the live wiring point between app.memory.{working,long_term,
policy} and the request path.

Working memory is pure in-process, mirroring app.approvals.store.ApprovalStore
(dict + lock, single-process, nothing here is meant to survive a restart).
Long-term memory is an in-memory index plus an append-only JSONL log,
mirroring app.observability.trace.TraceStore, so every SAVE/UPDATE/EXPIRE/
REJECT decision is durable, replayable evidence rather than a mutation of a
dict that vanishes on restart. REJECT/IGNORE decisions are logged too — a
poisoning rejection is exactly as important to have on record as a save.
"""

import threading
import time
from pathlib import Path

from pydantic import BaseModel

from app.memory.long_term import LongTermMemory, MemoryEntry
from app.memory.policy import MemoryCandidate, MemoryDecision, apply, decide, expire_stale
from app.memory.working import WorkingMemory
from app.rag.embed import EmbeddingProvider, NullEmbeddingProvider

MEMORY_LOG_PATH = Path("data/memory.jsonl")


class MemoryLogRow(BaseModel):
    timestamp: float
    user_id: str
    action: str  # MemoryAction value
    reason: str
    candidate_source: str
    candidate_subject: str
    memory_id: str | None = None


class MemoryStore:
    """Owns per-user_id WorkingMemory and LongTermMemory. `propose()` is the
    single call site every specialist's write hook goes through, so there is
    exactly one place a memory decision can end up un-logged."""

    def __init__(self, path: Path | str = MEMORY_LOG_PATH, embedder: EmbeddingProvider | None = None) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._working: dict[str, WorkingMemory] = {}
        self._long_term: dict[str, LongTermMemory] = {}
        self._embedder = embedder or NullEmbeddingProvider()

    # --- working memory (pure in-process, TTL, ApprovalStore-shaped) ------

    def get_working(self, user_id: str, *, now: float | None = None) -> WorkingMemory:
        with self._lock:
            wm = self._working.get(user_id, WorkingMemory())
            wm, _expired_keys = wm.expire_stale(now)
            self._working[user_id] = wm
            return wm

    def set_working(
        self, user_id: str, key: str, value: object, *, source: str = "assistant", ttl_s: float | None = None
    ) -> WorkingMemory:
        with self._lock:
            wm = self._working.get(user_id, WorkingMemory())
            kwargs = {"source": source}
            if ttl_s is not None:
                kwargs["ttl_s"] = ttl_s
            wm = wm.set(key, value, **kwargs)
            self._working[user_id] = wm
            return wm

    def clear_working(self, user_id: str, key: str) -> WorkingMemory:
        with self._lock:
            wm = self._working.get(user_id, WorkingMemory())
            wm = wm.clear(key)
            self._working[user_id] = wm
            return wm

    # --- long-term memory (in-memory index + JSONL append log) -----------

    def get_long_term(self, user_id: str) -> LongTermMemory:
        with self._lock:
            return self._long_term.get(user_id, LongTermMemory())

    def propose(self, user_id: str, candidate: MemoryCandidate, *, now: float | None = None) -> MemoryDecision:
        """decide() -> apply() -> append one durable log row. Every action
        (including REJECT/IGNORE, which leave the store unchanged) is logged,
        so a poisoning rejection is visible evidence, not a silent no-op."""
        with self._lock:
            memory = self._long_term.get(user_id, LongTermMemory())
            decision = decide(candidate, memory, now=now)
            vector = self._embed_if_available(candidate.text)
            updated = apply(decision, memory, vector=vector)
            self._long_term[user_id] = updated
            self._append_log(user_id, decision)
            return decision

    def recall(self, user_id: str, query: str, top_k: int = 3) -> list[MemoryEntry]:
        memory = self.get_long_term(user_id)
        return memory.recall(query, embedder=self._embedder, top_k=top_k)

    def expire_stale_long_term(self, user_id: str, *, now: float | None = None) -> list[MemoryDecision]:
        with self._lock:
            memory = self._long_term.get(user_id, LongTermMemory())
            updated, decisions = expire_stale(memory, now=now)
            self._long_term[user_id] = updated
            for decision in decisions:
                self._append_log(user_id, decision)
            return decisions

    def _embed_if_available(self, text: str) -> list[float] | None:
        if not self._embedder.available:
            return None
        vectors = self._embedder.embed_documents([text])
        return vectors[0] if vectors else None

    def _append_log(self, user_id: str, decision: MemoryDecision) -> None:
        row = MemoryLogRow(
            timestamp=time.time(),
            user_id=user_id,
            action=decision.action.value,
            reason=decision.reason,
            candidate_source=decision.candidate.source,
            candidate_subject=decision.candidate.subject,
            memory_id=decision.target_memory_id,
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(row.model_dump_json() + "\n")


def build_default_memory_store() -> MemoryStore:
    return MemoryStore()
