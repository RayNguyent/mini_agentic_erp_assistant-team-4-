# Runbook

## Local, no Docker (fastest path)

```bash
# 1. Backend deps
uv sync

# 2. Build the RAG index (BM25 always; Chroma vectors only if EMBEDDINGS_ENABLED=true)
uv run python -m app.rag.ingest --rebuild

# 3. Start the API (deterministic offline mode — no credential needed)
uv run uvicorn app.api.main:app --reload --port 8000

# 4. Frontend (separate terminal)
cd frontend && npm install && npm run dev
```

- API: http://127.0.0.1:8000 (docs at `/docs`, health at `/health`, readiness at `/readiness`)
- Frontend: http://localhost:3000
- Demo tokens (see `.env.example` → `AUTH_TOKENS_JSON` to override): `dev-token` (developer), `pm-token` (project_manager), `audit-token` (auditor) — select a role via the top-right sign-in control.

## Docker Compose

```bash
docker compose up --build
```

Builds and runs the API and frontend as two services (see `docker-compose.yml`).
The RAG index must still be built once before or after first boot — either
bake it into the image (`docker compose run api uv run python -m app.rag.ingest`)
or run it against the mounted `data/` volume.

## Health and readiness

```bash
curl localhost:8000/health       # {"status": "ok"} — liveness only
curl localhost:8000/readiness    # provider, model, RAG index status, degraded flag
```

`/readiness` reports `"degraded"` whenever the LLM provider is unconfigured
or the vector arm is unavailable — **this is expected and correct** in the
default deterministic/BM25-only profile, not a failure. It becomes `"ok"`
once `LLM_PROVIDER=openai` (or `ollama`) and `EMBEDDINGS_ENABLED=true` are
both set with a reachable provider.

## Running the test suite

```bash
uv run pytest -q                                    # ~350 backend tests
uv run python -m eval.runner                         # golden evaluation (writes eval/reports/)
cd frontend && npm run test:e2e                      # Playwright — requires npm install first
```

## Rebuilding the RAG index

Run after any change under `docs/corpus/`:

```bash
uv run python -m app.rag.ingest --rebuild
```

Writes `data/index/chunks.json` + `data/index/manifest.json` (BM25 source of
truth, always rebuilt) and, when `EMBEDDINGS_ENABLED=true`, replaces the
Chroma collection at `data/chroma/` wholesale — never an incremental upsert,
so a stale chunk from a previous corpus revision can never remain searchable
alongside the new ones. `--stats` reports without writing;
`data/ingestion_log.jsonl` accumulates one append-only record per run.

## Failure recovery / rollback

- **LLM provider outage**: transient errors retry with bounded, jittered
  backoff (`app/graph/retry.py`, `RetryPolicy(max_retries=2)`), then degrade
  to a typed `PROVIDER_ERROR` — the deterministic offline path is unaffected,
  since it makes no network call at all.
- **A write's tool call times out after dispatch**: `create_risk` is
  configured with `retry_limit=0` specifically so a timeout is never blindly
  retried; the idempotency-key reconciliation pattern for a real ERP backend
  is documented in `docs/odoo-mapping.md`.
- **Bad corpus content or a bad ingest run**: `--rebuild` is idempotent and
  safe to re-run; the previous `data/index/` is fully replaced, not merged.
- **Rollback**: this is a stateless service aside from `data/` (mock ERP
  fixtures, the RAG index, audit/trace logs) and the in-memory approval
  store. Rolling back to a previous commit/tag and restarting the process is
  sufficient; there is no schema migration to reverse.

## Configuration reference

See `.env.example` for the full list. Deterministic offline defaults
(`LLM_PROVIDER=deterministic`, `EMBEDDINGS_ENABLED=false`) require no
credential and are the default — every other value in the file is optional.
