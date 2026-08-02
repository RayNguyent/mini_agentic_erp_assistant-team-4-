# Final Design Document — Mini Agentic ERP Assistant

Required assessment evidence per `final-project.pdf` §3. Read alongside
`docs/architecture.md` (diagram) and `docs/adr/` (trade-off records).

## 1. Architecture diagram

See `docs/architecture.md`. Browser chat UI → FastAPI → identity/auth →
multi-agent graph runtime → {RAG, ERP tools, write approval} → LLM provider
adapters, with trace and audit capture on every request, all inside one
deployment/trust boundary (single process, in-memory approval store).

## 2. Service boundaries

Dependency direction is inward: UI → API → runtime → ports. Concrete
providers (`OpenAIProvider`, `OllamaProvider`, `MockERPProvider`,
`ChromaVectorIndex`) never leak into `app/agents/`, `app/runtime.py`, or
`app/graph/` — those modules depend only on the `Protocol`s in
`app/providers/base.py`, `app/providers/erp.py`, and `app/rag/vector_index.py`.
This is why swapping OpenAI↔Ollama↔deterministic, or Mock↔(future)Odoo, is a
`.env` change, not a code change — see `docs/architecture.md` § "Service
boundaries" for the specifics and ADR-004 for why the LLM adapter is a single
shared instance rather than one per consumer.

## 3. Typed reasoning state

`app/state.py::AgentState` — extended this pass with `request_id`, `actor`
(identity + role), `route` (`Route` enum: `tool` / `rag` / `multi_agent` /
`conversation` / `refusal`), `risk_level`, `required_context`,
`retrieved_sources: list[Citation]`, `plan: list[AgentStep]`,
`agent_results: dict[str, AgentResult]`, `context_log`, `idempotency_key`, on
top of the original `intent`, `selected_tool`, `tool_input/output`,
`approval_required`, `approved`, `retry_count`, `next_action`, `error_code`.
Five Pydantic validators make illegal states unconstructable rather than
merely checked at a boundary: a write's output can't exist without a recorded
approval; output and error can't both be set; `approved` can't be set unless
approval was required; and (new) a `route=RAG` answer can't exist without at
least one citation unless it's an explicit refusal.
`AgentResult` (one per specialist) and `AgentStep` (one per plan entry) are
the multi-agent-specific typed additions — the supervisor's plan and each
specialist's outcome are structured data, never free text the executor has
to re-parse.

## 4. Memory and context strategy

Three layers, each with a distinct policy (`app/memory/`):

- **Short-term** (`short_term.py`): the conversation buffer, allow-list
  compacted — a pending approval or refusal flag survives summarization
  verbatim; only unflagged turns get folded into the prose summary.
- **Working** (`working.py`): per-session task facts (active project code,
  in-flight approval, last tool result), TTL-expiring, explicit `expire_stale`
  sweep that reports which keys it removed rather than deleting silently.
- **Long-term** (`long_term.py`): durable semantic facts with provenance
  (`source`, `confidence`, `subject`) and staleness by age, recalled by
  substring match offline or cosine similarity when embeddings are available.

All three write through one policy gate (`policy.py::decide`): `SAVE`,
`UPDATE` (near-duplicate on the same subject, Jaccard-similarity based),
`IGNORE` (low confidence), `EXPIRE`, or `REJECT` — the reject path is the
memory-poisoning defense: a candidate from an untrusted source
(`document`/`tool_result`/`web`) matching a deterministic instruction-pattern
is never written, with the untrusted source named in the decision's reason
for provenance. Verified against the actual embedded-attack fixture in
`docs/corpus/vendor-intake-notes.md` (eval case `memory-11`).

The context builder (`app/context/builder.py`) assembles system policy, the
live user turn, retrieved evidence, tool results, and memory into one prompt
under an explicit token budget, admitting blocks in fixed priority order
(policy and the current turn always win) and logging every excluded block
with a reason (`ContextBuildLog.excluded`) — required evidence per the spec,
and the actual mechanism that keeps a budget overrun from silently dropping
the safety policy.

## 5. Model-provider strategy

Typed port (`app/providers/base.py::LLMProvider`): `generate` (JSON-mode),
`generate_text` (prose), `generate_tool_call`. Three adapters share this
contract: `default_classify` (deterministic, zero network, the offline
default), `OpenAIProvider`, and `OllamaProvider` (a thin subclass pointed at
Ollama's OpenAI-compatible `/v1` endpoint with no real credential required).
Selection is environment-driven (`ProviderSettings.from_env()`,
`LLM_PROVIDER=deterministic|openai|ollama`); credentials never appear in code
or logs (`ProviderSettings.redacted()`). Every adapter maps SDK-specific
exceptions onto the shared typed taxonomy (`ProviderError` = retryable,
`NonRetryableProviderError` = not) so retry logic can never loop on a bad API
key. Timeout is configurable per adapter (`timeout_s`); grounded-output
validation is the RAG citation gate (§7 below), which applies regardless of
which provider produced the draft answer; a provider failure degrades to the
deterministic path for classification/planning and to a typed error for
generation — it never silently drops the citation requirement.

## 6. ERP provider/plugin strategy

`app/providers/erp.py::ERPProvider` is the normalized port; `MockERPProvider`
is the required baseline, backing all 6 tools
(`get_project_status`, `list_project_tasks`, `get_sprint_progress`,
`get_budget_summary`, `list_risks`, `create_risk`). Field-level Odoo 19
mapping for every tool, installed-module assumptions, the external-identifier
strategy (never resolved by display name), and typed
`NOT_CONFIGURED`/`NOT_SUPPORTED` behavior are documented in full in
`docs/odoo-mapping.md` — no live `OdooERPProvider` is implemented (advanced
scope), and the risk register is explicitly documented as **not** an Odoo
core model, per the spec's architecture-accuracy requirement.

## 7. RAG / grounding strategy

`app/rag/agentic.py` — a bounded agentic loop (plan query → retrieve → grade
→ refine ≤3 attempts → generate → verify → done), not single-shot RAG.
Retrieval (`app/rag/retrieve.py`) fuses BM25 (pure Python, the offline
guarantee) and vector similarity (Chroma, embedded/local — ADR-002) via
Reciprocal Rank Fusion; ACL filtering (`app/rag/acl.py`) happens at recall,
before fusion, so a restricted chunk is never shown to the model to "choose"
not to repeat. Three independent gates stand between a retrieved chunk and a
shown answer, and only the first is a prompt: the prompt asks for citations;
`GroundedAnswer` cannot be constructed claiming sufficiency without one
(Pydantic validator); `verify_citations` strips any id the model was not
actually shown and demotes to refusal if nothing survives. Offline mode
(no LLM configured) uses extractive quoting of the top evidence with the same
citation contract — the golden eval suite exercises both paths identically.

## 8. Evaluation plan

`eval/golden/cases.v1.json` — 15 cases across 9 families (RAG+citation,
refusal, tool routing, multi-agent fan-out, bounded retry, approval
approve/reject, memory update, memory poisoning, authorization, document ACL,
prompt injection), each with an explicit `acceptance_rule` and machine-checked
`expect` conditions. `eval/runner.py` runs every case against the **real**
system (real tool registry, real RAG index, real permission checks — nothing
mocked) via 4 check kinds (`chat`, `retry`, `approval`, `memory_decision`) and
writes a versioned, timestamped report plus `eval/reports/latest.json`.
Thresholds: >80% overall pass rate (gate), 100% on the 6 cases tagged
`critical_case_ids` (approval approve/reject, memory poisoning, auth denial,
document ACL, prompt injection) — a `FAIL` on any critical case fails the run
regardless of the overall rate.

**A case genuinely failed during development and drove a real fix**, not a
test adjustment: eval case `retry-07` (bounded retry) failed with
`AttributeError: 'NoneType' object has no attribute 'max_retries'`. Root
cause: `run_multi_agent()`'s `retry_policy` parameter defaulted to `None`,
and `GraphContext.get("retry_policy", DEFAULT_RETRY_POLICY)` — a plain
`dict.get(key, default)` — only falls back to `default` when the key is
*absent*, not when its stored value is `None`; the multi-agent path had never
exercised an actual tool failure before this case, so the bug was latent.
Fixed in two places for defense-in-depth: the function's default is now
`DEFAULT_RETRY_POLICY` (not `None`), and the lookup site
(`app/agents/erp_analyst.py`) changed from `.get(key, default)` to
`.get(key) or default`, which is correct regardless of what a future caller
passes. A regression test
(`tests/test_agents.py::test_run_multi_agent_retries_a_transient_tool_failure_without_a_retry_policy_kwarg`)
locks the fix in independent of the eval suite. Full before/after and the
current report are in `eval/reports/`.

## 9. Quality and governance

Traces (`app/observability/trace.py`) capture one span per graph-node
transition plus richer domain spans (retrieval diagnostics, agent hops, tool
calls with retry counts, LLM calls, injection findings), keyed by
`request_id`, exported to `data/traces.jsonl` and queryable via
`GET /traces/{request_id}`. The audit log (`app/security/audit.py`) is a
separate append-only JSONL for auth, tool calls, approval decisions, blocked
actions, and final responses — critically, an approval's audit row is written
using the `approval_id` actually returned to the caller (a bug where the
runtime and the API layer each independently created a pending-approval
record, producing two different ids, was caught and fixed during this pass —
see `app/api/routes.py::_run_chat`'s comment). Prompt-injection controls are
deterministic pattern matching (`app/security/patterns.py`, shared between
request-time screening and the memory-write gate) — flagged, logged, but not
solely relied upon; the actual enforcement is structural (the approval gate,
the citation gate, per-tool permission checks) and holds even if a pattern is
missed. Residual risks — single-process state, static demo auth, pattern-only
injection screening, no rate limiting — are recorded in
`docs/threat-model.md` rather than left implicit.

## 10. Deployment and reliability

Single FastAPI process + Next.js dev server, or `docker compose up`
(`Dockerfile`, `frontend/Dockerfile`, `docker-compose.yml`) with a health
check gating frontend startup on backend readiness. Config and secrets flow
entirely through environment variables (`.env.example`); the deterministic
offline profile requires no credential and is the default. `/health` is
liveness-only; `/readiness` reports the active provider, model, RAG index
chunk/vector counts, and a `degraded` flag — degraded is the *expected* state
in the default profile, not a fault. Latency/cost budget and failure-recovery
mechanics (bounded jittered backoff, `retry_limit=0` on writes,
idempotency-key reconciliation pattern) are detailed in
`docs/runbook.md` and `docs/threat-model.md`. Rollback is a plain
revert-and-restart — no schema migration exists to reverse; `data/` (mock
fixtures, RAG index, audit/trace logs) is the only persisted state outside
version control.

## Trade-offs and ownership

Four ADRs recorded in `docs/adr/`: from-scratch graph runtime vs. LangGraph
(ADR-001), hybrid BM25+vector retrieval and Chroma vs. alternatives
(ADR-002), agentic RAG + supervisor/specialist topology vs. single-shot RAG
and vs. planner/executor/critic (ADR-003), and a single shared LLM provider
instance vs. the original per-consumer construction (ADR-004, a real
inconsistency found and fixed during this pass). Per-member component,
test, and evidence ownership is mapped in `docs/contribution-map.md`.
