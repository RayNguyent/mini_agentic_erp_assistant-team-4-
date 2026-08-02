# Contribution map

Extends the MVP ownership split in the original `README.md` "Team Assignments"
section onto the final-project components each area grew into. Every member
must be able to explain the end-to-end request path and answer questions
outside their primary component, per the assessment rubric.

## HoangNTV3 — Runtime & Routing → Graph runtime, typed state, multi-agent orchestration

| Component | Files | Tests |
|---|---|---|
| From-scratch graph engine | `app/graph/engine.py`, `app/graph/retry.py` | `tests/test_graph_engine.py` |
| Typed reasoning state | `app/state.py` (`AgentState`, `Route`, `Citation`, `AgentResult`, `ContextBuildLog`) | `tests/test_state.py` |
| Single-agent runtime | `app/runtime.py` | `tests/test_runtime.py`, `tests/test_end_to_end.py` |
| Multi-agent graph (supervisor/specialists/synthesizer) | `app/agents/` | `tests/test_agents.py` |
| Design decisions | ADR-001, ADR-003 | `docs/adr/` |

## BaoNG17 — Tools & Mock ERP → ERP tools, RAG layer, memory

| Component | Files | Tests |
|---|---|---|
| Extended `ERPProvider` (3 new tools) | `app/providers/erp.py`, `app/tools/erp_tools.py`, `app/tools/schemas.py` | `tests/test_erp_tools_extended.py`, `tests/test_tools.py`, `tests/test_providers.py` |
| Mock data fixtures | `data/tasks.json`, `data/milestones.json`, `data/budgets.json` | (exercised by the above) |
| Agentic RAG pipeline | `app/rag/` (ingest, chunk, bm25, embed, vector_index, retrieve, agentic, answer) | `tests/test_rag_retrieval.py`, `tests/test_rag_agentic.py`, `tests/test_rag_ingest.py` |
| Document corpus | `docs/corpus/*.md` | (ingested and retrieved in the above) |
| Adaptive memory + context builder | `app/memory/`, `app/context/` | `tests/test_memory.py`, `tests/test_context_builder.py` |
| Design decisions | ADR-002 | `docs/adr/` |
| Odoo mapping (design evidence) | `docs/odoo-mapping.md` | — |

## LamNH22 — FastAPI & Approval Workflow → API layer, security, observability

| Component | Files | Tests |
|---|---|---|
| API endpoints (chat/approve/reject/readiness/tools/traces) | `app/api/routes.py`, `app/api/schemas.py`, `app/api/dependencies.py`, `app/api/main.py` | `tests/test_api_chat.py`, `tests/test_api_approvals.py`, `tests/test_api_health.py`, `tests/test_api_security_and_rag.py` |
| Approval store | `app/approvals/` | `tests/test_approvals_store.py` |
| Auth, permissions, injection screening, audit log | `app/security/` | `tests/test_security.py` |
| Trace capture | `app/observability/` | `tests/test_observability.py` |
| Provider adapters (OpenAI, Ollama, config) | `app/providers/openai_provider.py`, `app/providers/ollama_provider.py`, `app/providers/config.py`, `app/providers/intent_classifier.py` | `tests/test_providers_openai.py`, `tests/test_providers_ollama.py`, `tests/test_intent_classifier.py` |
| Design decisions | ADR-004 | `docs/adr/` |
| Threat model, runbook | `docs/threat-model.md`, `docs/runbook.md` | — |

## MinhNDT6 — Testing & Documentation → Evaluation, browser e2e, frontend, docs

| Component | Files | Tests |
|---|---|---|
| Golden evaluation suite | `eval/golden/cases.v1.json`, `eval/runner.py` | `eval/reports/latest.json` (generated) |
| Browser chat UI extensions (citations, metadata, roles, a11y) | `frontend/app/page.tsx`, `frontend/components/`, `frontend/lib/api.ts` | `frontend/e2e/*.spec.ts` (Playwright) |
| Final Design Document, architecture diagram | `docs/final-design.md`, `docs/architecture.md` | — |
| This contribution map | `docs/contribution-map.md` | — |

## Cross-cutting (touched by more than one area, by necessity)

- `app/errors.py` — the typed error taxonomy every layer above shares.
- `app/tools/specs.py` — `TOOL_META` is read by both the tool gateway (LamNH22's area) and the ERP tool definitions (BaoNG17's area).
- `app/tokenizer.py` — used by both RAG chunking (BaoNG17) and the context builder (BaoNG17/HoangNTV3 boundary).

## Defense readiness

Every member should be able to walk the request path in
`docs/architecture.md` § "Request path" end to end — parse → plan → dispatch
→ retrieve/tool-call → synthesize → (approve) → trace/audit — regardless of
which specific module they own, since that path crosses all four areas in a
single request.
