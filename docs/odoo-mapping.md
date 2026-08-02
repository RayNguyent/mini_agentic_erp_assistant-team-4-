# Odoo 19 mapping (design evidence — no live adapter implemented)

This is required design evidence per the final-project spec: "the design
submission must also map that contract to Odoo 19... A live Odoo
implementation is advanced scope, but an inaccurate claim that a non-core
model is built into Odoo fails the architecture-accuracy review." No
`OdooERPProvider` exists in this repo; `MockERPProvider` (`app/providers/erp.py`)
is the required baseline every tool runs against, and it implements the same
`ERPProvider` port an Odoo adapter would.

## Integration contract

- **Endpoint**: External JSON-2 API, `POST /json/2/<model>/<method>`, bearer
  API key. Odoo Online requires a Custom plan for external API access; a
  local/self-hosted database must be checked against its own `/doc` page —
  installed modules, fields, and methods vary by database and version.
- **Service identity**: a dedicated bot user with minimum model access. JSON-2
  requests are evaluated through that user's access rights, record rules, and
  field access. The bot's access rights are **not** a substitute for the
  application-level authorization check in `app/security/permissions.py` —
  both must pass (see `docs/threat-model.md`).
- **External identifiers**: never resolved by matching on display name. The
  project code in every tool's public contract is not assumed to be a native
  Odoo field; the adapter resolves it through a configured custom field
  (`x_project_code`), an integration mapping table, or a validated numeric
  identifier.
- **Error normalization**: every adapter method maps Odoo/HTTP failures onto
  the typed `ErrorCode` set already used by `MockERPProvider`'s callers —
  `NOT_FOUND`, `FORBIDDEN`, `NOT_CONFIGURED`, `NOT_SUPPORTED`, `TIMEOUT`,
  `PROVIDER_ERROR` (`app/errors.py`). `NotConfiguredError` and
  `NotSupportedError` already exist in this codebase and are used today by
  `get_sprint_progress`/`get_budget_summary` against the mock provider —
  an Odoo adapter reuses the identical exception types.

## Tool-by-tool mapping

### `get_project_status(project_code)` — `project.read`

| Normalized field | Odoo 19 source |
|---|---|
| `project_code` | resolved via `x_project_code` custom field or mapping table, never by name |
| `name`, `stage`, `owner`, `status_summary` | `project.project`: `name`, `stage_id`, `user_id`, `last_update_status` |
| `start_date` / `end_date` | `project.project.date_start` / `date` |
| `open_task_count` / `task_count` | `project.project.open_task_count` / `task_count` |
| `blockers` | derived from authorized `project.task` records or an explicitly configured blocker rule — **never** a fabricated core field |

Mock equivalent: `data/projects.json` + `MockERPProvider.get_project`.

### `list_project_tasks(project_code, status?, limit?)` — `project.read`

Resolve the authorized `project.project`, then `search_read` scoped
`project.task` rows. Odoo task `state` values are normalized onto this
project's closed set of states (`open`, `in_progress`, `blocked`, `done`,
`cancelled`); `overdue` is derived with the exact rule this project already
implements in `app/providers/erp.py::is_overdue` — "a deadline in the past
**and** the task is not done or cancelled" — never a per-adapter
reinterpretation. A `blocked` filter must use documented task states or
configured stage rules, not title-text guessing.

Mock equivalent: `data/tasks.json` + `MockERPProvider.list_tasks`.

### `get_sprint_progress(project_code, iteration_ref?)` — `project.read`

Core Odoo Project does not guarantee a Scrum sprint object. The default
mapping profile — already named in this project's mock output as
`"milestone-as-iteration"` (`app/tools/schemas.py::SprintProgressOutput.mapping_profile`) —
treats `project.milestone` plus linked `project.task.milestone_id` as the
delivery iteration, using task `state` and `date_deadline` for
committed/completed/open/overdue counts. A Scrum add-on may instead map an
installed custom field such as `x_sprint_id`; whichever profile is active,
the response **must name it** — this project's mock tool already refuses to
silently label all project tasks as one sprint (`NotConfiguredError` when no
milestone data exists, `NotFoundError` for an unknown `iteration_ref`).

Mock equivalent: `data/milestones.json` + `MockERPProvider.list_milestones`.

### `get_budget_summary(project_code)` — `project.finance.read`

There is no universal single project-budget field in Odoo. An adapter reads
the project's `account_id` and only the installed sources configured for that
database — analytic lines, timesheets, sales, purchase, expenses, invoices —
combined the way Odoo's profitability view does. Preferred implementation:
one audited custom read method (e.g. `agent.project_budget_snapshot`) so the
snapshot is computed in a single JSON-2 transaction rather than the adapter
assembling several independent calls that could observe an inconsistent
state mid-write. If a planned budget or a source module is absent, the
adapter returns `NOT_CONFIGURED` — this project's mock tool already does
exactly that (`app/providers/erp.py::require_budget_source`) rather than
reporting a zero balance, and `data/budgets.json` deliberately includes a
project with no entry at all (`PRJ-003`) and one with every figure null
(`PRJ-002`) to keep that path under test.

Mock equivalent: `data/budgets.json` + `MockERPProvider.get_budget`.

### `list_risks(project_code, state?)` — `project.risk.read`

Odoo Project has **no portable core risk-register model**. An Odoo adapter
requires a custom add-on or Studio model — e.g. `x_project_risk` — with a
`Many2one` link to `project.project` and documented fields
(title, description, probability, impact, severity, owner, state, due date).
`search_read` under explicit access rights and project/company record rules.
If the custom model is not installed on the target database, the adapter
returns `NOT_SUPPORTED` (already a distinct typed outcome from
`NOT_CONFIGURED` in `app/errors.py` — a capability gap, not a missing
setting). **The risk register must never be represented as an Odoo core
model** — that misrepresentation is explicitly called out in the spec as an
architecture-accuracy failure.

Mock equivalent: `data/risks.json` + `MockERPProvider.list_risks`.

### `create_risk(project_code, risk_payload, idempotency_key)` — `project.risk.create` + human approval

After approval (enforced in `app/agents/risk_writer.py` / `app/runtime.py::execute_write_tool`
before the tool gateway is ever reached), an Odoo adapter would call **one**
custom method — e.g. `x_project_risk/action_create_from_agent` — that
resolves the project, validates allowed values, enforces company/project
scope, checks the idempotency key, creates the record, and returns the
normalized result, all inside one JSON-2 transaction. A generic unrestricted
`create` call is insufficient: each JSON-2 call is its own transaction, so a
multi-step create (resolve → validate → write) done as separate calls could
observe or leave inconsistent state on a timeout between them. A timeout
after dispatch must be reconciled by an idempotency-key lookup before any
retry — write tools in this project are already configured with
`retry_limit=0` for exactly this reason (`app/tools/specs.py::TOOL_META["create_risk"]`);
retrying a write is never done blindly.

Mock equivalent: `MockERPProvider.create_risk` (`app/providers/erp.py`), driven
through the approval gate in `app/runtime.py::execute_write_tool` and
`app/agents/risk_writer.py::RiskWriter`.

## Installed-module assumptions (for a future live adapter)

- Project, Milestones (core) — required for `get_project_status`, `list_project_tasks`, `get_sprint_progress`.
- Analytic Accounting, and whichever of Timesheets/Sales/Purchase/Expenses/Accounting are actually enabled — required for `get_budget_summary`; absent modules degrade to `NOT_CONFIGURED` per-source, not a hard failure.
- A custom Studio/add-on model (`x_project_risk`) — required for `list_risks`/`create_risk`; absent entirely degrades to `NOT_SUPPORTED`.
- A custom field or mapping table for the external project code (`x_project_code`) — required for every tool; never resolved by display name.

## What is required design evidence vs. what remains unbuilt

Required and present: this document (field-level mapping, module
assumptions, typed unsupported/not-configured behavior, external-identifier
strategy) and the shared `ERPProvider` port both `MockERPProvider` and a
future `OdooERPProvider` would implement identically.

Not built (advanced/bonus scope, explicitly out of scope for this pass):
a running `OdooERPProvider`, a live target-database `/doc` snapshot, a
provider-contract test run against a real Odoo instance, and the Docker
Compose Odoo/PostgreSQL runbook. A shared provider-contract test suite
already exists conceptually — `MockERPProvider` is exercised by
`tests/test_providers.py` and `tests/test_erp_tools_extended.py` against the
same normalized tool contract an Odoo adapter would need to satisfy; adding
`OdooERPProvider` would run those same test functions parameterized over the
new adapter rather than writing a second test suite.
