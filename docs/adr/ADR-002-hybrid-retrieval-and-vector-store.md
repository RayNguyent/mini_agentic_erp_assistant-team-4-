# ADR-002: Hybrid BM25 + vector retrieval, Chroma as the vector store

## Status
Accepted

## Context
The RAG layer is the project's stated differentiator and must satisfy the
spec's hard gate ("no working project-document RAG with citations: core
project requirement is not met") while also supporting the deterministic
offline profile: "Deterministic offline defaults must require no credential."
A pure-embedding pipeline cannot meet both — an embedding call always requires
a configured provider.

## Decision
Retrieval fuses two independent arms with Reciprocal Rank Fusion (RRF):

1. **BM25** (`app/rag/bm25.py`), implemented from scratch, pure Python, zero
   dependencies, zero network calls. This is the offline guarantee.
2. **Vector similarity** via an embedding provider (OpenAI-compatible or
   Ollama) behind an `EmbeddingProvider` port, indexed in **Chroma**
   (embedded, local-persist mode — no server) behind a `VectorIndex` port.

RRF combines the two by *rank*, not by raw score, so the vector arm can be
absent entirely (unconfigured embeddings) without needing to renormalize
anything — `NullEmbeddingProvider`/`NullVectorIndex` make "no vector arm" a
first-class, always-tested code path (`app/rag/retrieve.py::Retriever.uses_vectors`).

## Alternatives considered

**Pure embeddings, no keyword arm.** Rejected: breaks the offline/no-credential
requirement outright, and empirically weaker on this corpus's exact
identifiers (`PRJ-001`, `RISK-2`, `429`, currency figures) where keyword match
is the stronger signal.

**Vector store: Chroma vs. a hand-rolled numpy matrix vs. Qdrant (Docker) vs.
LanceDB.**
- A numpy matrix + JSON sidecar (the original draft of this ADR) is
  dependency-free and exact at this corpus's scale (single-digit documents,
  low hundreds of chunks), but reads as "not a real vector database" for a
  reviewer and has no metadata-filtered query — ACL filtering would have to
  happen entirely after recall.
- Qdrant via Docker Compose is the most production-shaped choice and would
  make the deployment diagram more realistic, but it makes the offline demo
  and the Playwright/e2e run depend on Docker being up for *any* RAG query,
  which conflicts directly with "deterministic mode always works offline."
- **Chroma (embedded, persistent, chosen)** is a real vector database with
  metadata `where`-filtering (used for the ACL pre-filter,
  `app/rag/vector_index.py::ChromaVectorIndex.search`), but runs in-process
  with a local directory — no server dependency, so it does not compromise the
  offline guarantee (it is simply unused/empty when embeddings are disabled).
- LanceDB was considered as a middle ground (embedded, columnar) but is less
  familiar to a reviewer and offers no capability Chroma lacks for this
  project's needs.

## Consequences
- ACL filtering happens in two layers: Chroma's `where` clause narrows the
  vector-arm candidates by classification before they are even returned, and
  `app/rag/acl.py::filter_chunks` re-filters after fusion regardless — defence
  in depth, not reliance on the index query alone.
- The retriever degrades visibly, not silently: `RetrievalDiagnostics.degraded`
  and a note are set whenever the vector arm is unavailable, surfaced in
  `/readiness` and every retrieval trace.
- Re-ingesting the corpus (`python -m app.rag.ingest --rebuild`) resets and
  rewrites the Chroma collection wholesale rather than upserting incrementally,
  so a stale chunk from a previous corpus revision can never remain searchable.
