"""Hybrid retrieval: BM25 + vectors, fused by Reciprocal Rank Fusion.

Why RRF rather than a weighted score blend: BM25 scores are unbounded and
corpus-dependent while cosine scores sit in [-1, 1], so any fixed weighting is
really a hidden calibration that drifts as the corpus changes. RRF only consumes
*ranks*, so the two systems combine without either needing to be normalised
against the other, and adding or removing the vector arm cannot silently
rescale the keyword arm.

Order of operations matters and is deliberate:

    recall (bm25 ∥ vector) -> ACL filter -> fuse -> optional rerank -> top-k

ACL filtering happens at recall, before anything is fused or shown to a model.
"""

import logging
import time
from dataclasses import dataclass, field

from app.rag.acl import clearance_for, filter_chunks
from app.rag.bm25 import BM25Index
from app.rag.documents import Chunk, ScoredChunk
from app.rag.embed import EmbeddingProvider, NullEmbeddingProvider
from app.rag.store import ChunkStore
from app.rag.vector_index import NullVectorIndex, VectorIndex
from app.state import Citation, Role

logger = logging.getLogger(__name__)

RRF_K = 60  # standard damping constant; larger = flatter rank contribution
DEFAULT_TOP_K = 5
RECALL_K = 20


@dataclass(slots=True)
class RetrievalDiagnostics:
    """Per-query evidence: which arm fired, what ACL removed, how long it took.

    Exported into the trace and rendered in the UI's debug drawer. This is the
    "retrieval diagnostics" artefact the assessment asks for.
    """

    query: str
    role: str | None = None
    bm25_hits: int = 0
    vector_hits: int = 0
    fused_candidates: int = 0
    acl_denied: list[str] = field(default_factory=list)
    reranked: bool = False
    returned: int = 0
    elapsed_ms: float = 0.0
    degraded: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "query": self.query,
            "role": self.role,
            "bm25_hits": self.bm25_hits,
            "vector_hits": self.vector_hits,
            "fused_candidates": self.fused_candidates,
            "acl_denied_count": len(self.acl_denied),
            "acl_denied": self.acl_denied,
            "reranked": self.reranked,
            "returned": self.returned,
            "elapsed_ms": round(self.elapsed_ms, 1),
            "degraded": self.degraded,
            "notes": self.notes,
        }


@dataclass(slots=True)
class RetrievalResult:
    results: list[ScoredChunk]
    diagnostics: RetrievalDiagnostics

    @property
    def is_empty(self) -> bool:
        return not self.results

    def citations(self) -> list[Citation]:
        return [to_citation(scored) for scored in self.results]

    def context_blocks(self) -> list[str]:
        return [
            f"[{s.chunk.chunk_id}] {s.chunk.title} › {s.chunk.heading}\n{s.chunk.text}"
            for s in self.results
        ]


def to_citation(scored: ScoredChunk) -> Citation:
    chunk = scored.chunk
    snippet = chunk.text.strip().replace("\n", " ")
    return Citation(
        doc_id=chunk.doc_id,
        chunk_id=chunk.chunk_id,
        title=chunk.title,
        heading=chunk.heading,
        snippet=snippet[:280] + ("…" if len(snippet) > 280 else ""),
        score=round(scored.score, 4),
        classification=chunk.classification,
    )


def _rrf(rank: int) -> float:
    return 1.0 / (RRF_K + rank + 1)


class Retriever:
    """Owns the index and the fusion policy for one corpus."""

    def __init__(
        self,
        store: ChunkStore,
        embedder: EmbeddingProvider | None = None,
        vector_index: VectorIndex | None = None,
        *,
        top_k: int = DEFAULT_TOP_K,
        recall_k: int = RECALL_K,
    ) -> None:
        self.store = store
        self.embedder = embedder or NullEmbeddingProvider()
        self.vector_index = vector_index or NullVectorIndex()
        self.top_k = top_k
        self.recall_k = recall_k
        self._bm25 = BM25Index(
            [f"{c.title} {c.heading} {c.text}" for c in store.chunks]
        )
        self._index_of = {c.chunk_id: i for i, c in enumerate(store.chunks)}

    @property
    def size(self) -> int:
        return self.store.size

    @property
    def uses_vectors(self) -> bool:
        return self.embedder.available and self.vector_index.available

    def _vector_recall(
        self,
        query: str,
        role: Role | str | None,
        diagnostics: RetrievalDiagnostics,
    ) -> list[tuple[int, float]]:
        if not self.uses_vectors:
            diagnostics.degraded = True
            diagnostics.notes.append(
                "vector arm unavailable — keyword-only retrieval (citations still enforced)"
            )
            return []
        try:
            hits = self.vector_index.search(
                self.embedder.embed_query(query),
                top_k=self.recall_k,
                # Push the ACL into the index query itself. The retriever
                # re-filters below regardless; this is the first of two gates.
                allowed_classifications=sorted(clearance_for(role)),
            )
        except Exception as exc:  # noqa: BLE001 - degrade to BM25, never fail the query
            logger.warning("vector recall failed, falling back to BM25: %s", exc)
            diagnostics.degraded = True
            diagnostics.notes.append(f"vector recall failed: {type(exc).__name__}")
            return []

        # Map chunk ids back to store positions so both arms fuse on one key.
        return [
            (self._index_of[chunk_id], score)
            for chunk_id, score in hits
            if chunk_id in self._index_of
        ]

    def retrieve(
        self,
        query: str,
        *,
        role: Role | str | None = None,
        top_k: int | None = None,
        project_code: str | None = None,
        rerank=None,
    ) -> RetrievalResult:
        started = time.perf_counter()
        limit = top_k or self.top_k
        diagnostics = RetrievalDiagnostics(query=query, role=str(role) if role else None)

        if not self.store.chunks:
            diagnostics.notes.append("index is empty — run `python -m app.rag.ingest`")
            diagnostics.elapsed_ms = (time.perf_counter() - started) * 1000
            return RetrievalResult([], diagnostics)

        bm25_hits = self._bm25.search(query, top_k=self.recall_k)
        vector_hits = self._vector_recall(query, role, diagnostics)
        diagnostics.bm25_hits = len(bm25_hits)
        diagnostics.vector_hits = len(vector_hits)

        # --- ACL: drop what this role may not read, before fusion ----------
        allowed_index: dict[int, Chunk] = {}
        denied: set[str] = set()
        for index in {i for i, _ in bm25_hits} | {i for i, _ in vector_hits}:
            chunk = self.store.chunks[index]
            visible, blocked = filter_chunks([chunk], role)
            if visible:
                allowed_index[index] = chunk
            denied.update(blocked)
        diagnostics.acl_denied = sorted(denied)

        # Optional metadata narrowing. Applied after ACL so a project filter can
        # never widen what a role is allowed to see.
        if project_code:
            allowed_index = {
                i: c
                for i, c in allowed_index.items()
                if c.project_code in (project_code, "ALL")
            }

        # --- Fuse -----------------------------------------------------------
        fused: dict[int, ScoredChunk] = {}
        for rank, (index, score) in enumerate(bm25_hits):
            if index not in allowed_index:
                continue
            fused[index] = ScoredChunk(
                chunk=allowed_index[index],
                score=_rrf(rank),
                bm25_score=round(score, 4),
                bm25_rank=rank + 1,
                matched_by=["bm25"],
            )
        for rank, (index, score) in enumerate(vector_hits):
            if index not in allowed_index:
                continue
            existing = fused.get(index)
            if existing is None:
                fused[index] = ScoredChunk(
                    chunk=allowed_index[index],
                    score=_rrf(rank),
                    vector_score=round(score, 4),
                    vector_rank=rank + 1,
                    matched_by=["vector"],
                )
            else:
                fused[index] = existing.model_copy(
                    update={
                        "score": existing.score + _rrf(rank),
                        "vector_score": round(score, 4),
                        "vector_rank": rank + 1,
                        "matched_by": [*existing.matched_by, "vector"],
                    }
                )

        candidates = sorted(fused.values(), key=lambda s: -s.score)
        diagnostics.fused_candidates = len(candidates)

        if rerank is not None and candidates:
            try:
                candidates = rerank(query, candidates)
                diagnostics.reranked = True
            except Exception as exc:  # noqa: BLE001 - rerank is an optimisation
                logger.warning("rerank failed, using fused order: %s", exc)
                diagnostics.notes.append(f"rerank failed: {type(exc).__name__}")

        results = candidates[:limit]
        diagnostics.returned = len(results)
        diagnostics.elapsed_ms = (time.perf_counter() - started) * 1000
        return RetrievalResult(results, diagnostics)


def build_retriever(
    index_dir=None,
    embedder: EmbeddingProvider | None = None,
    vector_index: VectorIndex | None = None,
) -> Retriever:
    """Assemble a retriever from configuration, degrading rather than failing.

    A missing index, an unconfigured embedding provider, or an unavailable
    Chroma directory each narrow what retrieval can do; none of them are
    startup errors.
    """
    from app.providers.config import ProviderSettings
    from app.rag.embed import build_embedding_provider
    from app.rag.store import INDEX_DIR
    from app.rag.vector_index import build_vector_index

    settings = ProviderSettings.from_env()
    store = ChunkStore.load(index_dir or settings.index_dir or INDEX_DIR)
    embedder = embedder or build_embedding_provider(settings)
    if vector_index is None:
        vector_index = build_vector_index(enabled=embedder.available)
    return Retriever(store, embedder, vector_index)
