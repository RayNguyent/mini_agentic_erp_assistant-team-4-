"""Long-term semantic memory: durable facts, each with provenance.

Reuses the embedding port from app.rag so long-term memory and document
retrieval share one adapter contract rather than two. Every entry records
where it came from and how confident the writer was — provenance is what lets
`policy.py` later explain *why* a stale or poisoned entry was rejected, instead
of just deleting it silently.
"""

import time
import uuid

from pydantic import BaseModel, Field

from app.rag.embed import EmbeddingProvider, NullEmbeddingProvider

DEFAULT_STALE_AFTER_S = 90 * 24 * 60 * 60  # 90 days


class MemoryEntry(BaseModel):
    memory_id: str
    text: str
    written_at: float
    source: str  # "user_turn" | "tool_result" | "document" | "agent_inference"
    confidence: float = 1.0
    subject: str = ""  # e.g. a project_code, for scoped recall
    vector: list[float] = Field(default_factory=list)

    def stale(self, now: float | None = None, max_age_s: float = DEFAULT_STALE_AFTER_S) -> bool:
        return (now or time.time()) - self.written_at > max_age_s


class LongTermMemory(BaseModel):
    entries: list[MemoryEntry] = Field(default_factory=list)

    def add(self, entry: MemoryEntry) -> "LongTermMemory":
        return self.model_copy(update={"entries": [*self.entries, entry]})

    def update(self, memory_id: str, text: str) -> "LongTermMemory":
        updated = [
            e.model_copy(update={"text": text, "written_at": time.time()}) if e.memory_id == memory_id else e
            for e in self.entries
        ]
        return self.model_copy(update={"entries": updated})

    def remove(self, memory_id: str) -> "LongTermMemory":
        return self.model_copy(update={"entries": [e for e in self.entries if e.memory_id != memory_id]})

    def stale_entries(self, now: float | None = None) -> list[MemoryEntry]:
        return [e for e in self.entries if e.stale(now)]

    def for_subject(self, subject: str) -> list[MemoryEntry]:
        return [e for e in self.entries if e.subject == subject]

    def recall(self, query: str, embedder: EmbeddingProvider | None = None, top_k: int = 3) -> list[MemoryEntry]:
        """Cosine recall when embeddings are available, substring match
        otherwise — long-term memory degrades the same way document retrieval
        does, rather than going silent without a configured provider."""
        embedder = embedder or NullEmbeddingProvider()
        if embedder.available and self.entries:
            return self._vector_recall(query, embedder, top_k)
        lowered = query.lower()
        matches = [e for e in self.entries if lowered in e.text.lower()]
        return matches[:top_k]

    def _vector_recall(self, query: str, embedder: EmbeddingProvider, top_k: int) -> list[MemoryEntry]:
        import numpy as np

        query_vec = np.asarray(embedder.embed_query(query), dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm == 0:
            return []
        query_vec = query_vec / norm

        scored = []
        for entry in self.entries:
            if not entry.vector:
                continue
            vec = np.asarray(entry.vector, dtype=np.float32)
            vec_norm = np.linalg.norm(vec)
            if vec_norm == 0:
                continue
            score = float(np.dot(query_vec, vec / vec_norm))
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda pair: -pair[0])
        return [entry for _, entry in scored[:top_k]]


def new_memory_id() -> str:
    return f"MEM-{uuid.uuid4().hex[:10]}"
