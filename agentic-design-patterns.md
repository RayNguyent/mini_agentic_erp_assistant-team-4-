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
