# Design Patterns & Production Concepts — Agentic ERP Assistant Course

## Architecture

- **Ports & Adapters (Hexagonal Architecture)** — define abstract interfaces (`Protocol`/ABC) for things like LLM provider, ERP provider, retrieval, memory. Business logic depends only on the interface, never a vendor SDK. This is *why* swapping OpenAI↔Ollama or Mock↔Odoo doesn't touch your agent code.
  → `week-01/module-02-.../lab-src/src/unit02_architecture/ports.py`, `week-02/module-04-.../src/assistant/llm/ports.py`

- **Dependency Injection / Composition Root** — concrete adapters (DB session, LLM client) are only wired together in one place (app startup / FastAPI `Depends`), never inside domain code.
  → `week-01/module-03-.../lab-src/src/unit03_service_shell/app.py` (`get_document_service`, etc.)

- **Repository pattern** — persistence (SQLAlchemy) is hidden behind a repository class/port so services never touch `session.add` directly.
  → `unit03_service_shell/repository.py`, `ports.py`

- **Decorator / higher-order function for cross-cutting concerns** — e.g. `@observed` wraps service methods to record metrics without polluting business logic. Same idea as middleware.
  → `unit01_foundation/registry.py` (`register_parser`), `unit03_service_shell/observability.py` (`@observed`)

- **Typed state machine for agent reasoning** — instead of letting the LLM's free text drive control flow, every reasoning step produces a validated Pydantic object (route, confidence, required tool, approval flag). This makes agent behavior testable and auditable instead of "vibes from a prompt."
  → `week-02/module-05-.../src/assistant/reasoning/decision.py` (`ReasoningDecision`)

## LLM-specific production concerns

- **Structured outputs / schema-constrained generation** — force the model to return JSON matching a Pydantic schema (with cross-field validators) instead of parsing free text. Stops hallucinated "grounded" answers with no citations.
  → `week-02/module-04-.../src/assistant/llm/schemas.py` (`GroundedAnswer`)

- **Role-separated prompting** — system / developer / user / evidence roles kept separate so retrieved (untrusted) content can never be confused with an instruction. This is the core prompt-injection defense.
  → `week-02/module-04-.../src/assistant/llm/prompts.py`

- **Real token counting + context budget** — use `tiktoken` to measure exact tokens, not character-count guesses, and refuse to send a request that would exceed the model's context window.
  → `week-02/module-05-.../src/assistant/context/builder.py`, `tokenizer.py`

- **Context compaction as an allow-list** — when summarizing long conversations, explicitly whitelist which fields survive (pending approvals, safety flags) instead of a generic "summarize everything" that can silently drop safety-critical state.
  → `week-02/module-05-.../src/assistant/context/compact.py`

- **Router / intent classification pattern** — a cheap, structured LLM call decides *which* path to take (retrieve docs / call tool / ask approval / clarify / refuse) before the expensive generation call runs. Prevents wasted cost and unsafe default behavior.
  → `week-02/module-05-.../src/assistant/reasoning/router_client.py`, `router_runner.py`

- **Model-policy-as-config** — which model/temperature/token-budget each route uses lives in a reviewed YAML allowlist, not hardcoded in Python, so upgrading a model is a config change with an audit trail.
  → `config/model-policy.yaml`, `src/assistant/reasoning/model_policy.py`

- **Exponential backoff + jitter retry** — only retry *transient* errors (timeouts, 429, 5xx), never a broken prompt or bad auth; jitter avoids synchronized retry storms against the provider.
  → `week-02/module-04-.../src/assistant/llm/retry.py`

- **Typed error taxonomy** — distinguish `TransientProviderError` / `ProviderAuthError` / `ClientConfigurationError` so retry logic can never accidentally loop forever on a bad API key.
  → `week-02/module-04-.../src/assistant/llm/ports.py`

- **Real cost/telemetry accounting** — price every call from a dated, per-model rate table; raise instead of silently guessing a price for an unknown model.
  → `week-02/module-04-.../src/assistant/llm/pricing.py`, `telemetry.py`

## Safety / governance

- **Human-in-the-loop approval gate** — any side-effecting action (e.g. `create_risk`) must be held pending until explicit human approval, logged with an actor and correlation id. Required pattern for any agent that can *write*, not just read.
  → final-project spec §7 "Approval gate"; final project `create_risk` tool contract

- **Idempotency keys for write actions** — a retried/timed-out write must be safely reconciled instead of duplicating a side effect.
  → final-project spec, `create_risk(project_code, risk_payload, idempotency_key)`

- **Audit trail / trace envelope** — every event (tool call, approval, retrieval) carries actor, correlation id, and timestamp so behavior is replayable after the fact.
  → `week-01/module-02-.../lab-src/src/unit02_architecture/events.py`

- **Untrusted-content isolation** — retrieved documents, tool outputs, and web results are treated as data, never instructions (ties back to role-separated prompting above).

## RAG

- **Citation-required grounding + explicit refusal** — the model must cite a retrieved source id for any claim, and refuse/ask for clarification when it can't. Enforced at the schema level, not just prompted for.

- **ACL filtering before generation, not after** — restricted document chunks must be excluded from the retrieval set the model sees; never rely on the model to "not mention" content it was already shown.
  → final-project spec §2 RAG layer, §6 quality gate "RAG quality"

## MCP-style tool boundary

- **Typed tool contract, not a direct function call** — every tool declares input schema, output schema, permission level, side-effect level (read vs write), timeout, and retry limit. The agent calls the tool boundary, never business logic or a provider SDK directly from a reasoning node. This is what makes "swap MockERPProvider for OdooERPProvider without touching the agent" possible.
  → final-project spec §2 "MCP-style tool boundary", §6 "forbidden shortcuts: do not call provider-specific ERP code directly from agent nodes"

- **Tool registry as the single source of truth** — one place lists every tool's schema, permission, and audit category, so both the router (deciding which tool to call) and the security gate (deciding whether it's allowed) read from the same definition instead of duplicating rules.
  → `app/tools/registry.py`, `app/tools/specs.py`

- **Normalized provider-error taxonomy** — provider failures collapse to a small typed set (`NOT_FOUND`, `FORBIDDEN`, `NOT_CONFIGURED`, `NOT_SUPPORTED`, `TIMEOUT`, `PROVIDER_ERROR`) instead of leaking raw vendor exceptions up through the tool boundary. Callers branch on the typed outcome, not on string-matching an error message.
  → final-project spec §9 "Odoo adapter acceptance rules"

## ERP / external-system provider plugin

- **Shared contract tests across adapters** — `MockERPProvider` and a real `OdooERPProvider` must both pass the *same* test suite against the same normalized `ERPProvider` port. If a mock-only test suite exists, the abstraction isn't actually verified.
  → final-project spec §2 "ERP provider plugin", §9 "Mock and Odoo adapters must pass the same provider contract tests"

- **Don't invent fields the underlying system doesn't guarantee** — map only to fields that are confirmed to exist in the target system's schema/docs (e.g. Odoo's `/doc` page for that specific database); return `NOT_CONFIGURED`/`NOT_SUPPORTED` rather than fabricating a plausible-looking value. A wrong claim that a non-core model exists is treated as an architecture-accuracy failure, not a minor bug.
  → final-project spec §9 "Required tool scenarios" (e.g. `get_budget_summary`, `list_risks`)

- **One transactional method per side-effecting action** — a write tool should resolve identifiers, validate values, enforce scope, check idempotency, and persist, all inside a single provider-side transaction/method — not as several separate calls the agent orchestrates, which could partially fail.
  → final-project spec §9 `create_risk` mapping ("call one custom method... in one transaction")

- **Service-identity ≠ application authorization** — a shared bot/service credential used to talk to the external system is a *separate* concern from checking whether the requesting user is allowed to see the result. Checking the bot's access rights is not a substitute for checking the human's.
  → final-project spec §7 "Odoo service identity"

## Adaptive memory

- **Three distinct memory layers, not one blob** — short-term conversation state, working project/task memory, and long-term semantic memory each have separate write/update/expire/compaction rules. Collapsing them into "the chat history" loses the ability to reason about staleness or provenance.
  → final-project spec §2 "Adaptive memory"

- **Provenance-tagged memory writes** — every stored memory item records where it came from (user turn, tool result, retrieved doc) so a later "stale" or "poisoned" memory can be identified and rejected with evidence, not guesswork.
  → final-project spec, minimum demo scenario "Memory policy"

- **Untrusted memory candidates** — content proposed for long-term memory (especially anything sourced from tool output or web search) is validated before being written, the same way retrieved documents are treated as untrusted before being cited.
  → final-project spec §7 "External-content isolation"

## Evaluation & observability

- **Golden test suite as regression gate, not a one-time report** — versioned cases (≥10) each name expected route, expected source/tool behavior, and an acceptance rule up front; a case that starts failing after a change is a regression, not "the model changed its mind."
  → final-project spec §2 "Evaluation and observability", §6 "Evaluation" quality gate (>80% overall, 100% on auth/approval/secret-handling cases)

- **Trace = replayable causal chain, not just a log line** — one trace links prompt, retrieved chunk ids, tool calls, retries, latency, and the final route decision under one correlation id, so a failure can be replayed end-to-end instead of reconstructed from scattered logs.
  → final-project spec §2 "Evaluation and observability", §8 "Trace package"

- **A failed golden case must change the implementation, not the test** — the evaluation plan explicitly requires a narrative of "which case failed → root cause → what was fixed," so the eval suite drives design decisions instead of being adjusted to match whatever the agent currently does.
  → final-project spec §3 "Evaluation plan", defense question 7

## Reliability & deployment

- **Bounded retry, and only for safe operations** — retries apply to transient errors on reads or on writes that are idempotent/reconciled by key; a non-idempotent write is never blindly retried, since a timeout doesn't mean the operation didn't happen.
  → final-project spec §2 "Reliability boundary", §9 "Timeout after dispatch must be reconciled by idempotency lookup before retry"

- **Deterministic offline mode as the safety net** — the LLM provider port must have a mode that requires no credential and always works, so the whole system degrades to something demonstrable instead of hard-failing when a real provider is unreachable.
  → final-project spec §2 "LLM provider boundary", §8 "Deterministic offline defaults must require no credential"

- **Health/readiness as a first-class endpoint** — readiness reports which provider/model is actually active (not just "service is up"), so a demo or an operator can distinguish "healthy but degraded to fallback" from "fully configured."
  → final-project spec, minimum demo scenario "Configured model provider"

- **Secrets never touch the browser or the trace payload** — API keys stay in server-side config; demo tokens the UI lets a user pick live in memory only, never in browser storage, URLs, rendered messages, screenshots, or trace/audit records.
  → final-project spec §7 "Browser credential handling", §7 "Secrets and sensitive telemetry"

## Architecture Decision Records

- **ADRs compare rejected alternatives, not just justify the chosen one** — a defensible ADR states what else was considered (e.g. "why not LangGraph as the orchestrator") and why it was rejected, so a design choice can be defended under questioning instead of asserted.
  → final-project spec §3 "Trade-offs and ownership", §6 rubric "Architecture design and trade-off defense" (20% weight)

- **Framework-boundary discipline** — orchestration frameworks (LangGraph, CrewAI, AutoGen, etc.) may be studied or benchmarked but must not implement the core agent loop, memory manager, tool router, retry engine, or graph runtime — those stay hand-rolled so the team can explain and defend every control-flow decision.
  → final-project spec §1, §6 "Forbidden shortcuts"

## Production tips & guidelines checklist

- Prefer returning a typed `NOT_CONFIGURED`/`NOT_SUPPORTED` outcome over guessing a value — a wrong-but-confident answer is worse than an honest gap, especially for financial/ERP data.
- Treat every external input the same way regardless of source (retrieved doc, tool result, web search, memory candidate): untrusted until validated, never confused with an instruction.
- Any side-effecting tool call is approval-gated and idempotent by construction — write the idempotency key and audit row *before* attempting the write, not after.
- Keep model/route/temperature/budget choices in reviewed config, not scattered literals, so a provider or model change is a diffable, auditable change.
- A component isn't "done" until it has a negative/failure test (invalid schema, denied permission, timeout, expired memory) — the happy path alone doesn't demonstrate the boundary actually holds.
