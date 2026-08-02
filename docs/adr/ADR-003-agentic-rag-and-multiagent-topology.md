# ADR-003: Agentic RAG loop, and supervisor + specialists as the multi-agent topology

## Status
Accepted

## Context
Two related decisions: (1) how retrieval itself behaves — a single
retrieve-then-answer pass, or an iterating agentic loop; (2) how work is
divided across agents for a request that may need ERP data, project
documents, or both.

## Decision

### Agentic RAG (single retrieval pass would not suffice)
`app/rag/agentic.py` implements a bounded loop: plan query → retrieve → grade
→ refine (bounded, ≤3 attempts) → generate → verify citations → done. Every
arm is a node in the same from-scratch graph engine as the rest of the
project (ADR-001), so the iteration is traced with real per-attempt
diagnostics, not hidden inside one opaque retrieval call. The step limit — not
a prompt — is what guarantees termination.

### Supervisor + 3 specialists + synthesizer
`app/agents/graph.py`: a supervisor produces a typed plan
(`list[AgentStep]`); `ERPAnalyst` (5 read tools), `DocResearcher` (the agentic
RAG loop above), and `RiskWriter` (the sole path to `create_risk`) execute it;
a synthesizer merges results, unions citations, and only reports a top-level
error when every specialist failed.

## Alternatives considered

**Single-shot RAG** (retrieve once, answer). Rejected: fails on
conversational phrasing that shares no vocabulary with the corpus ("is it
going to be late?" vs. the corpus's "go-live date", "milestone", "schedule")
and on compound questions needing evidence no single query ranks together.
Confirmed in practice — `grade_deterministic`'s per-chunk relevance floor
exists precisely because a single BM25 pass over the raw question returns
thin, off-topic evidence for those cases (see `tests/test_rag_agentic.py`).

**Planner → Executor → Critic.** Considered for the multi-agent topology:
stronger reflection story, but a heavier LLM-call budget per turn and a less
natural mapping onto this project's actual capability boundaries (ERP tools
vs. documents vs. writes are a permission split, not a plan/execute/critique
split). The agentic RAG loop already provides the "critic" role for the
document path specifically (the verify step), without paying the cost for
every ERP-tool-only request.

**Single-agent loop with a bigger tool list** (no specialists at all — what
the codebase looked like before this work, minus the 3 new tools). Simplest
to reason about, but the permission story becomes a single flat allowlist
instead of a structural confinement: nothing stops a future tool addition
from being reachable by every request path. Specialist allowlists
(`ERPAnalyst.allowed_tools`, `DocResearcher.allowed_tools = frozenset()`,
`RiskWriter.allowed_tools`) make "the document researcher cannot call
create_risk" true by construction, not by convention.

## Consequences
- A request needing both ERP data and document context (e.g. "what's the
  status of PRJ-001 and what does the risk policy say about ownership?") gets
  a fan-out plan and a synthesized answer with both a tool result and a
  citation in one turn (`tests/test_agents.py::test_run_multi_agent_*`,
  eval case `multiagent-06`).
- Latency cost: a fan-out plan makes two tool/retrieval calls instead of one.
  Acceptable at this project's scale; would need explicit parallelization
  (currently sequential in `_run_agents_node`) if specialist count grew.
- The approval-suspend point had to be made resumable *mid-plan* (a write
  step can appear after other steps have already completed) — solved by
  keeping completed results in `state.agent_results` across the suspend/resume
  boundary (`app/agents/graph.py::_run_agents_node`).
