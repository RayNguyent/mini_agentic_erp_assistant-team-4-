# ADR-004: One shared LLM provider instance, not one per consumer

## Status
Accepted

## Context
The pre-existing codebase constructed two separate `OpenAIProvider` instances
— one in `get_default_classifier()` (used for intent classification) and one
in `get_llm_provider()` (used for conversational fallback) — each independently
reading `ProviderSettings.from_env()` and building its own `openai.OpenAI`
client. Adding the supervisor's planning call and the RAG loop's grounding
call would have made this four separate client instances for what is
configured as one provider.

## Decision
`app/api/dependencies.py::get_llm_provider()` is now the single
`@lru_cache`-d source of the LLM adapter; `get_intent_classifier()` is built
from the same cached `ProviderSettings` (`_shared_provider_settings()`) and
constructs its classifier consistently with it. The supervisor
(`app/agents/supervisor.py::plan`) and the RAG answerer
(`app/rag/answer.py::answer_from_documents`) both accept a `provider` argument
from the same dependency rather than constructing their own.

## Alternatives considered

**Leave the asymmetry** (each consumer builds its own client). Rejected —
confirmed as a real inconsistency during this work: a config change (e.g.
switching `LLM_PROVIDER=ollama`) had to be correctly threaded through every
independent construction site, and it is exactly the kind of drift that
produces "the classifier is on OpenAI but generation somehow isn't" bugs in
production.

**A global singleton module-level variable** instead of a FastAPI dependency.
Rejected: breaks test isolation (`tests/conftest.py` overrides
`get_llm_provider` per test via `app.dependency_overrides`) and breaks the
evaluation harness, which constructs its own `Harness` independent of the
FastAPI app entirely (`eval/runner.py::Harness`).

## Consequences
- One `.env` change updates every LLM-consuming path (classification,
  planning, RAG grounding, conversational fallback) consistently.
- `OllamaProvider` (a thin `OpenAIProvider` subclass, `app/providers/ollama_provider.py`)
  slots into the same single dependency without any consumer needing to know
  which concrete adapter is active.
- Cost/latency accounting (a stated production concern) now has one place to
  instrument per-call telemetry rather than four.
