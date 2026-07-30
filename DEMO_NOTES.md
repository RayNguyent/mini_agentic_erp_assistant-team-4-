# Demo Talking Points

## Open with (15s)
- Tool-calling agent over a mock ERP
- Typed **state machine**, not a prompt chain — every step is `AgentState → AgentState`
- 3 tools: `get_project_status`, `list_risks` (read) · `create_risk` (write, **approval-gated**)

---

## Live flow — say this at each step

**1. `What's the status of PRJ-001?`**
- Classifier → intent → registry → tool. Read tools run immediately, no gate
- Status events stream over SSE (you see the agent's decisions, not just the answer)

**2. `add a risk`**
- No project, no title, no severity — **form appears instantly anyway**
- *This is the interesting part →* see "Two schemas" below

**3. Fill title + severity → Approve**
- `/chat` never wrote anything. It parked the state; `/approve` resumed it
- Approve button is disabled until required fields are filled

**4. `Show risks for PRJ-001`**
- New risk is there — write actually landed

**5. `add a risk` → Reject**
- `APPROVAL_REJECTED`, nothing written

**6. `What's the status of PRJ-999?`**
- Clean `NOT_FOUND` error, not a 500 — tool errors become responses

---

## Two schemas for one tool (your best point)
- **`CreateRiskDraftInput`** → given to the LLM. Everything optional
- **`CreateRiskInput`** → gates execution. Strict, rejects blank titles
- **Why:** strict schema → LLM has nothing valid to call on *"add a risk"* → it declines → falls back to a chat reply asking *"what's the title? severity?"*
- Lenient schema → calls the tool immediately → **form appears instead of an interrogation**
- Strictness lives at execution time, where it belongs. Never trust the form

---

## Architecture points (drop naturally)
- **HITL gate:** writes are *structurally* incapable of running unapproved — `WRITE_TOOLS` → `AWAIT_APPROVAL` → store
- **Approval ids `pop`** — resolve exactly once, no replay
- **Protocols everywhere** (`ERPProvider`, `LLMProvider`, `ToolRegistry`) — swap mock ERP → Odoo with zero runtime change
- **`_evolve()`** rebuilds via constructor so validators re-run on *every* transition
- **Invariants in `AgentState`:** write output can't exist without `approved=True`; output and error mutually exclusive
- **Errors:** Retryable (auto-retry, max 2) vs NonRetryable (fail fast)
- **Degrades gracefully:** no API key → deterministic keyword classifier, still demoable offline

---

## If asked about limits — own these
- In-memory store → **single uvicorn worker**, restart wipes state
- Approvals never expire
- Small models often skip title/severity extraction — harmless, form collects them
- 12 pre-existing test failures, unrelated: classifier returns `unsupported` on provider error where tests expect a `default_classify` fallback + `list_risks` seed-data drift

---

## Don't forget
- Show **`/docs`** (Swagger) — typed contracts for free
- Point at the **status lines** in the UI — that's the agent reasoning, streamed
- Backend `:8000` · Frontend `:3000` · both running before you start
