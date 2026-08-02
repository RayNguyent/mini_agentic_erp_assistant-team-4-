# Mini Agentic ERP Assistant (team 4)

## Overview

A Mini Agentic ERP Assistant for project delivery operations: a browser chat
interface backed by a from-scratch multi-agent graph runtime, project-document
RAG with citations, six ERP tools, approval-gated writes, three-layer memory,
role-based auth, and a golden evaluation suite — all built from scratch per
the final-project constraints (no LangGraph/CrewAI/AutoGen as orchestrator).

**Start here for the full picture:**
[Final Design Document](docs/final-design.md) ·
[Architecture diagram](docs/architecture.md) ·
[ADRs](docs/adr/) ·
[Threat model](docs/threat-model.md) ·
[Runbook](docs/runbook.md) ·
[Odoo mapping](docs/odoo-mapping.md) ·
[Contribution map](docs/contribution-map.md)

Capabilities, briefly:

- **Multi-agent runtime** (`app/agents/`): a supervisor plans a typed step
  list; `ERPAnalyst`, `DocResearcher`, and `RiskWriter` execute it, each
  confined to its own tool allowlist and permission; a synthesizer merges
  results and unions citations. See ADR-001, ADR-003.
- **Project-document RAG** (`app/rag/`): a bounded agentic retrieve → grade →
  refine → verify loop over hybrid BM25 + Chroma-vector retrieval, ACL-filtered
  by role before fusion. A grounded answer cannot exist without a real
  citation — enforced by a Pydantic validator, not a prompt. See ADR-002.
- **ERP tools** (`app/tools/`, `app/providers/erp.py`): `get_project_status`,
  `list_project_tasks`, `get_sprint_progress`, `get_budget_summary`,
  `list_risks`, `create_risk` (approval-gated), against `MockERPProvider`; a
  field-level Odoo 19 mapping exists as design evidence (`docs/odoo-mapping.md`).
- **Memory** (`app/memory/`): short-term (allow-list compacted), working
  (TTL-expiring session facts), long-term (provenance-tagged, with a
  deterministic memory-poisoning rejection gate).
- **Security** (`app/security/`): bearer-token auth, a role×permission
  matrix, deterministic prompt-injection screening, an append-only audit log.
- **Observability** (`app/observability/`): one trace per request, spanning
  every node transition, retrieval, agent hop, and tool call.
- **Evaluation** (`eval/`): 15 golden cases across 9 families, run against the
  real system — `uv run python -m eval.runner`.

---

# Running the App

See [docs/runbook.md](docs/runbook.md) for the full reference (Docker, health
checks, index rebuilds, failure recovery). Quick start:

```bash
uv sync
uv run python -m app.rag.ingest --rebuild   # build the RAG index (BM25 always; offline, no credential)
uv run uvicorn app.api.main:app --reload --port 8000

# separate terminal
cd frontend && npm install && npm run dev
```

- API: http://127.0.0.1:8000 (docs `/docs`, health `/health`, readiness `/readiness`)
- UI: http://localhost:3000 — sign in via the top-right role selector (demo tokens: `dev-token` / `pm-token` / `audit-token`) before chatting; every request requires an identity.

With both running, try the [demo scenarios](#demo-scenarios) below. Write
actions (like `create_risk`) surface an inline Approve/Reject card before
running, and a document question (e.g. "what does the risk policy say about
severity ratings?") returns cited source cards from `docs/corpus/`.

## Running tests

```bash
uv run pytest                       # ~350 backend tests
uv run python -m eval.runner        # golden evaluation suite
cd frontend && npm run test:e2e     # Playwright (requires npm install first)
```

---

# ERP Tools

## Read

```python
get_project_status(project_code)
list_project_tasks(project_code, status=None, limit=None)
get_sprint_progress(project_code, iteration_ref=None)
get_budget_summary(project_code)          # requires project.finance.read
list_risks(project_code)
```

## Write (approval-gated)

```python
create_risk(project_code, risk_payload)   # requires project.risk.create + human approval
```

---

# Architecture

See [docs/architecture.md](docs/architecture.md) for the full diagram and
request-path walkthrough. Summary:

```text
Browser (role-based sign-in, SSE, citations)
  │
  ▼
FastAPI  →  Auth (bearer → Actor{role})
  │
  ▼
Multi-agent graph runtime (supervise → run_agents → synthesize)
  │
  ├── ERPAnalyst ──► Tool gateway ──► MockERPProvider
  ├── DocResearcher ──► Agentic RAG (BM25 ∥ Chroma → RRF → ACL → cite)
  └── RiskWriter ──► approval gate ──► MockERPProvider
  │
  ▼
Trace + audit capture (every request)
```

---

# Project Structure

```text
project-root/
│
├── app/
│   ├── api/            FastAPI routes, schemas, dependencies (auth wired here)
│   ├── graph/           from-scratch graph engine + bounded retry (app.graph.engine.Graph)
│   ├── runtime.py       single-agent graph: intent → route → tool/conversation
│   ├── agents/          supervisor + specialists (ERPAnalyst, DocResearcher, RiskWriter) + synthesizer
│   ├── rag/              ingest, chunk, bm25, embed, vector_index (Chroma), retrieve, agentic loop, answer
│   ├── memory/           short-term / working / long-term + the save/update/ignore/expire/reject policy
│   ├── context/          token-budgeted context assembly
│   ├── security/         auth, permissions, injection screening, audit log
│   ├── observability/    trace capture
│   ├── tools/            ERP tool implementations, schemas, MCP-style boundary metadata
│   ├── providers/        LLMProvider/ERPProvider adapters (deterministic, OpenAI, Ollama, Mock)
│   ├── approvals/        approval store
│   └── state.py          AgentState and every typed model the runtime shares
│
├── data/                mock ERP fixtures + generated RAG index (gitignored) + audit/trace logs (gitignored)
├── docs/
│   ├── corpus/           the 9 project documents RAG ingests
│   ├── adr/               Architecture Decision Records
│   ├── final-design.md, architecture.md, threat-model.md, runbook.md, odoo-mapping.md, contribution-map.md
├── eval/                golden dataset + runner
├── frontend/            Next.js chat UI + Playwright e2e specs
└── tests/               ~350 backend tests
```

---

# Core Components

## Agent State

```python
class AgentState(BaseModel):
    intent: str
    selected_tool: str | None
    tool_input: dict | None
    tool_output: dict | None
    approval_required: bool = False
    approved: bool | None = None
    retry_count: int = 0
    next_action: str
```

---

## Provider Abstraction

```python
class LLMProvider(Protocol):
    def generate(self, prompt: str) -> str:
        ...
```

### Providers

```text
DeterministicProvider
OllamaProvider (future)
OpenAIProvider (future)
```

---

## Runtime Flow

```text
START
  │
  ▼
Parse Intent
  │
  ▼
Route Decision
  │
  ├── Read Tool
  │       │
  │       ▼
  │   Format Response
  │
  └── Write Tool
          │
          ▼
      Approval
          │
          ▼
      Execute Tool
          │
          ▼
      Format Response

          ▼
         END
```

---

> The sections below record the original MVP planning. For how ownership maps
> onto the final system's actual modules, see [docs/contribution-map.md](docs/contribution-map.md).

# Team Assignments

## 👨‍💻 HoangNTV3: Runtime & Routing

### Responsibilities

Build the core agent workflow and routing logic.

### Tasks

#### Typed State

- Create `AgentState`
- Add validation rules
- Define enums/constants if necessary

#### Runtime

Implement:

```text
parse_intent

route_decision

execute_read_tool

execute_write_tool

format_response
```

#### Retry Logic

```text
Maximum retries: 2
```

### Deliverables

```text
app/runtime/
app/state/
```

### Success Criteria

- Correct routing
- Typed state usage throughout runtime
- Retry logic works
- Unit tests pass

---

## 👨‍💻 BaoNG17: Tools & Mock ERP

### Responsibilities

Build business logic and mock data layer.

### Tasks

#### Mock Data

Create:

```text
data/projects.json
data/risks.json
```

#### ERP Provider

Implement:

```python
ERPProvider
MockERPProvider
```

#### Tool Schemas

```python
ProjectStatusInput
ProjectStatusOutput

RiskInput
RiskOutput
```

#### Tool Implementations

```text
get_project_status()

list_risks()

create_risk()
```

### Deliverables

```text
app/tools/
app/providers/
data/
```

### Success Criteria

- Tools return correct results
- Provider abstraction works
- Mock ERP data supports all tool scenarios
- Contract tests pass

---

## 👨‍💻 LamNH22: FastAPI & Approval Workflow

### Responsibilities

Expose the agent through APIs and manage approval flow.

### Tasks

#### Chat Endpoint

```http
POST /chat
```

Request:

```json
{
  "message": "What's the status of PRJ-001?"
}
```

Response:

```json
{
  "answer": "...",
  "tool_used": "get_project_status"
}
```

#### Approval Endpoints

```http
POST /approve

POST /reject
```

#### Health Check

```http
GET /health
```

### Deliverables

```text
app/api/
app/approvals/
```

### Success Criteria

- API endpoints work
- Approval flow works
- Health endpoint available

---

## 👨‍💻 MinhNDT6: Testing & Documentation

### Responsibilities

Ensure system quality and maintain documentation.

### Tasks

#### Golden Tests

Create test cases for:

```text
Project status lookup

List risks

Create risk approved

Create risk rejected

Unsupported request

Retry scenario
```

#### Documentation

Maintain:

- README
- Setup guide
- API usage guide
- Architecture overview
- Demo instructions

### Deliverables

```text
tests/
docs/
README.md
```

### Success Criteria

- Test coverage for all scenarios
- Documentation is up to date
- Demo instructions are clear

---

# Demo Scenarios

## Scenario 1: Project Status

### Request

```text
What's the status of PRJ-001?
```

### Expected Behavior

```text
Agent routes to get_project_status()
Returns project details
```

---

## Scenario 2: Create Risk

### Request

```text
Create a risk for PRJ-001
```

### Expected Behavior

```text
Approval Required

Approve

Risk Created
```

---

## Scenario 3: List Risks

### Request

```text
Show all risks for PRJ-001
```

### Expected Behavior

```text
Agent routes to list_risks()
Returns associated risks
```

---

## Scenario 4: Unsupported Request

### Request

```text
What's the weather today?
```

### Expected Behavior

```text
Request rejected

No hallucinated answer
```

---

## Scenario 5: Grounded Document Question (RAG)

### Request

```text
What are the risk severity ratings and who can close a risk?
```

### Expected Behavior

```text
Routes to DocResearcher (agentic RAG loop)
Answer cites a real chunk id from docs/corpus/risk-management-policy.md
Source card shown in the UI with title, heading, and snippet
```

## Scenario 6: Multi-Agent Fan-Out

### Request

```text
What is the status of PRJ-001 and what does the risk policy say about ownership?
```

### Expected Behavior

```text
Supervisor plans both erp_analyst and doc_researcher
Synthesized answer contains both a tool result and a citation
```

## Scenario 7: Role-Scoped Permission Denial

### Request (signed in as Developer)

```text
What is the budget for PRJ-001?
```

### Expected Behavior

```text
error_code: FORBIDDEN — developer lacks project.finance.read
Signed in as Project Manager, the same question succeeds
```

---

# Known Limitations & Bugs Encountered

Recorded honestly rather than smoothed over — several of these were caught by
the golden evaluation suite or by deliberate end-to-end checking, not by
inspection alone, and the fixes are what actually make the eval suite's
"failed case changed the implementation" evidence real
(see [docs/final-design.md](docs/final-design.md) §8).

## Bugs found and fixed during development

1. **Infinite loop in the RAG retrieval loop.** `refine → verify → refine`
   could cycle forever once refinement ran out of new queries to try, because
   neither node incremented the retry `attempt` counter the loop's budget was
   keyed on. Caught by the graph engine's step-limit guard (`StepLimitExceeded`),
   not by inspection. Fixed with an explicit `exhausted` flag that forces the
   loop to a terminal decision instead of bouncing.

2. **The grounding grader was too permissive.** "How many staff work in the
   Tokyo office?" was answered from unrelated sprint-plan text instead of
   refusing, because sufficiency was judged on aggregate keyword coverage
   across all retrieved chunks combined. Fixed by requiring at least one
   individual chunk to clear its own relevance floor — an aggregate score
   dragged over the line by several weakly-related chunks no longer counts.

3. **`RiskWriter` used stale, pre-approval input.** After a human edited the
   risk draft (title/severity/description) and approved it, the write used
   the *original* plan step's frozen input instead of the form-edited values
   merged in by the API layer — approving a corrected draft would have
   silently written the uncorrected one. Fixed by reading `state.tool_input`
   (the post-merge value) once a decision (`approved`) exists, and the
   original plan input only before that.

4. **`/chat` created two separate pending-approval records for one write.**
   Both the internal chat-handling function and its caller called
   `store.create(state)`, minting two different `approval_id`s — one leaked
   in the approval store forever, and the audit log recorded the wrong id
   entirely (not the one actually returned to the client). Fixed by making
   approval-record creation happen exactly once, in the caller, with the
   audit row keyed to that same id.

5. **Per-tool permission checks were agent-level, not tool-level.** `ERPAnalyst`
   wraps five tools that don't all require the same permission — `get_budget_summary`
   needs `project.finance.read`, the other four need only `project.read` — but
   the authorization check used one fixed permission for the whole agent. A
   `developer` asking for budget data would have been silently allowed through
   the agent gate. Caught by an end-to-end API test that actually sent the
   request as a `developer` and asserted `403 FORBIDDEN`, not by a unit test
   that only ever exercised `project.read` paths. Fixed by resolving the
   specific tool first, then checking *its* declared permission.

6. **Multi-agent retry silently broke.** `run_multi_agent(retry_policy=None)`
   stored an explicit `None` in the per-request context; the downstream
   lookup used `ctx.get("retry_policy", DEFAULT_RETRY_POLICY)`, and
   `dict.get(key, default)` only falls back to `default` when the key is
   **absent**, not when its stored value is `None` — so any transient tool
   failure crashed with `AttributeError: 'NoneType' object has no attribute
   'max_retries'` instead of retrying. Invisible until a case actually
   exercised a failing tool: caught by golden eval case `retry-07`, which
   deliberately injects a tool that fails twice before succeeding. Fixed in
   two places for defense-in-depth — the function's default is now a real
   `RetryPolicy`, and the lookup site no longer relies on `dict.get`'s
   default-only-on-missing-key behavior. Locked in with a regression test
   independent of the eval suite.

7. **RAG evidence had no token budget before it reached the LLM.** A
   token-budgeted context assembler (`app/context/builder.py`) was built and
   unit-tested in isolation, but was never actually imported anywhere outside
   its own module — the real prompt-construction path
   (`app/rag/answer.py::build_evidence_prompt`) just concatenated every
   retrieved chunk with no token accounting at all, relying only on an
   incidental chunk-*count* cap (`top_k=5`) that happened to stay safe at
   this corpus's current chunk sizes. Found when asked directly whether RAG
   context budgeting existed — confirmed by grepping for actual call sites,
   not by re-reading the module in isolation. Fixed by wiring
   `build_evidence_prompt` through `app.context.builder`, returning the
   included/excluded `ContextBuildLog` alongside the prompt; citations are
   now verified only against chunks that survived the budget, not every
   retrieved chunk — a chunk the budget excluded was never shown to the
   model, so citing it must be treated exactly like citing an id that was
   never retrieved. Verified end-to-end against the real corpus: at a tight
   budget, weaker-ranked but smaller chunks get admitted over larger
   higher-ranked ones exactly as the stable rank-order admission is supposed
   to do. One secondary finding from that verification, not fixed: each
   evidence block's `<block id="..." document="..." section="...">` wrapper
   repeats the full document title and section heading as attributes, adding
   30–50% token overhead over the raw chunk text — invisible at the default
   2000-token budget (nothing gets excluded on this corpus), but real at a
   tighter one.

8. **`Docker Compose` volume mount would have shadowed the baked-in RAG index.**
   The original `./data:/app/data` mount would silently replace the index
   built into the image at `docker build` time with an empty directory on a
   fresh clone, since `data/index/` and `data/chroma/` are gitignored. Found
   by static review (Docker isn't available in the environment this project
   was built in, so this was never actually run) — fixed by dropping that
   mount; audit/trace logs are ephemeral per-container in the default compose
   file rather than persisted via a fragile single-file bind mount, which has
   its own footgun (Docker creates a missing bind-mount *file* target as a
   *directory*, which would break the audit log's `open(path, "a")`).

## Known limitations (not fixed — flagged instead of hidden)

- **The frontend and Playwright e2e specs are written but not verified.**
  No Node.js/npm was available in the environment this project was built in,
  so `npm install`, `next build`, `eslint`, and `playwright test` were never
  actually run against this code. The API contract they call
  (`frontend/lib/api.ts`) is verified end-to-end from the backend side
  (`tests/test_api_security_and_rag.py` exercises the same request/response
  shapes), but the frontend code itself needs a real `npm install && npm run
  test:e2e` pass before it should be treated as confirmed working.
- **Docker is written but not built or run**, for the same reason (no Docker
  daemon in the build environment). The compose/Dockerfiles received a
  careful static review (see bug #7 above) but not an actual `docker compose
  up --build`.
- **Real token-level LLM streaming is not implemented.** `/chat/stream`
  completes the full multi-agent run (RAG grounding and citation
  verification have to finish before an answer is safe to show) and then
  replays the finished answer word-by-word over SSE — a deliberate,
  documented simplification (`app/api/routes.py::_stream_response`), not
  true incremental token streaming from the provider.
- **Single-process, in-memory state.** The approval store and the audit
  log's in-memory index assume one worker process; a multi-worker deployment
  would fragment or lose that state. Documented in
  [docs/threat-model.md](docs/threat-model.md) as a residual risk, not fixed.
- **No live Odoo adapter.** Only the field-level mapping design evidence
  exists ([docs/odoo-mapping.md](docs/odoo-mapping.md)); `MockERPProvider` is
  the only implemented `ERPProvider`, per the assignment's required baseline.
