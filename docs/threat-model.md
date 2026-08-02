# Threat model

## Assets

- Project documents (some `finance`/`restricted` classification), ERP data (project status, tasks, risks, budget), the risk register (writable), audit/trace logs, LLM/embedding provider credentials, demo bearer tokens.

## Actors and trust boundaries

| Actor | Trust level | Enters via |
|---|---|---|
| Authenticated user (`developer` / `project_manager` / `auditor`) | Trusted identity, scoped permissions | `Authorization: Bearer <token>` → `app/security/auth.py::TokenAuthenticator` |
| Unauthenticated caller | Untrusted | Any request without a valid bearer token — rejected with 401 before reaching the runtime (`app/api/dependencies.py::get_current_actor`) |
| Retrieved document content | Untrusted data, never an instruction source | `app/rag/answer.py::build_evidence_prompt` (explicit `<evidence untrusted="true">` wrapper) |
| Tool output | Untrusted data | Rendered by deterministic Python formatters (`app/runtime.py::_TOOL_RENDERERS`), never fed back into a prompt as free text |
| LLM provider | Semi-trusted — can misbehave, cannot bypass structural gates | Behind `LLMProvider`/`EmbeddingProvider` ports; every adapter maps failures to typed errors, never raw SDK exceptions |

## Permission matrix

Source of truth: `app/security/permissions.py::PERMISSIONS`.

| Permission | developer | project_manager | auditor |
|---|:---:|:---:|:---:|
| `project.read` | ✅ | ✅ | ✅ |
| `project.risk.read` | ✅ | ✅ | ✅ |
| `project.risk.create` | ❌ | ✅ | ❌ |
| `project.finance.read` | ❌ | ✅ | ✅ |
| `document.read` (base) | ✅ | ✅ | ✅ |

Document classification adds a second, independent scale
(`app/rag/acl.py::_CLEARANCE`):

| Classification | developer | project_manager | auditor |
|---|:---:|:---:|:---:|
| `public` | ✅ | ✅ | ✅ |
| `internal` | ✅ | ✅ | ✅ |
| `finance` | ❌ | ✅ | ✅ |
| `restricted` | ❌ | ❌ | ✅ |

An unauthenticated/unknown actor (`role=None`) gets the narrowest of both
scales — no tool permission, `public`-only document clearance — never the
widest, matching least-privilege default in both `app/security/permissions.py::check`
and `app/rag/acl.py::clearance_for`.

## Threats and mitigations

| # | Threat | Mitigation | Evidence |
|---|---|---|---|
| T1 | Caller reads/writes without an identity | Every route requires a bearer token; `AuthError` → 401 | `app/api/main.py::auth_error_handler`, `tests/test_api_security_and_rag.py::test_chat_without_a_token_is_unauthorized` |
| T2 | A role reads data outside its permission (e.g. `developer` reading budget) | Per-**tool** permission check before the tool boundary — not just a per-agent check, since `ERPAnalyst` wraps 5 tools with different permissions | `app/agents/erp_analyst.py`, `tests/test_api_security_and_rag.py::test_developer_is_denied_budget_summary` |
| T3 | A role sees a document above its classification | ACL filtering happens at **recall**, before fusion, before the prompt — the model is never shown a restricted chunk to "choose not to repeat" | `app/rag/acl.py`, `app/rag/retrieve.py::Retriever.retrieve`, eval case `acl-13` |
| T4 | A retrieved document tries to override system policy ("ignore all previous instructions", "skip approval", "reveal your API key") | (a) Explicit untrusted-evidence delimiter, separating instruction from data structurally; (b) deterministic pattern screening logs the attempt (`app/security/injection.py`); (c) the actual gates (approval, citation) are enforced in code, never by reading the model's compliance | `app/security/patterns.py`, eval case `injection-14`, `docs/corpus/vendor-intake-notes.md` (a real embedded-attack fixture) |
| T5 | A malicious/poisoned fact is written to long-term memory | `app/memory/policy.py::decide` rejects instruction-shaped candidates from untrusted sources before they are ever written — checked before confidence/duplicate logic | `tests/test_memory.py::test_document_sourced_instruction_like_content_is_rejected_not_saved`, eval case `memory-11` |
| T6 | A write executes without a human decision | `AgentState`'s own Pydantic validator makes "tool_output set on an approval-required action without `approved=True`" an unconstructable state — not just a runtime check | `app/state.py::AgentState.tool_output_requires_approval` |
| T7 | A retried write duplicates a side effect | `create_risk` has `retry_limit=0` (never blindly retried); idempotency reconciliation is documented as the pattern for a real ERP adapter (`docs/odoo-mapping.md`) | `app/tools/specs.py::TOOL_META["create_risk"]` |
| T8 | An approval decision is lost if execution fails right after | Audit row is written for the approval outcome as part of the same request that resolves it, independent of whether the subsequent tool call succeeds | `app/api/routes.py::_resume` |
| T9 | A model fabricates a citation it was never shown | `verify_citations` strips any id not in the retrieved set; if nothing survives, the answer is demoted to a refusal | `app/rag/answer.py::verify_citations`, `tests/test_rag_agentic.py::test_a_provider_answer_citing_nothing_it_was_shown_becomes_a_refusal` |
| T10 | Secrets leak into prompts/traces/screenshots | `.env` values only; `ProviderSettings.redacted()` reports presence, never the credential value; demo tokens live in browser JS memory only, never `localStorage`/cookies | `app/providers/config.py::redacted`, `frontend/lib/api.ts` (module-scoped `authToken`, no persistence), `frontend/e2e/acl-and-auth.spec.ts::"a session never persists a credential"` |
| T11 | Provider outage cascades into user-visible failure | Deterministic offline mode has no external dependency; a live provider's transient failure retries with bounded, jittered backoff, then degrades to a typed error rather than hanging | `app/graph/retry.py`, `app/errors.py` |

## Residual risks (not mitigated in this pass)

- **Single-process, in-memory state.** `ApprovalStore`, `AuditLog`'s in-memory
  mirror, and the trace store's index all assume one uvicorn worker. A
  multi-worker deployment would lose pending approvals on worker restart and
  fragment audit/trace visibility across workers. Documented, not fixed —
  out of scope for this project's deployment target.
- **Demo auth is a static token map**, not a real identity provider. Adequate
  for the assignment's "basic authentication, role checks" baseline; not
  production-grade session management (no expiry, no rotation, no
  revocation list beyond editing `AUTH_TOKENS_JSON`).
- **Injection screening is pattern-based**, not a learned classifier —
  documented in `app/security/patterns.py` as a floor, not a complete
  defense. A novel phrasing not in the pattern list would not be flagged,
  though it still could not bypass the structural gates (approval, citation)
  it would need to defeat to cause harm.
- **No rate limiting** on any endpoint in this pass.
