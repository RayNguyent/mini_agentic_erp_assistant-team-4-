# ADR-001: From-scratch graph runtime instead of LangGraph

## Status
Accepted

## Context
The assistant needs a control-flow engine that routes a request through
intent parsing, tool execution, retry, an approval suspend/resume point, and
(for the multi-agent path) a supervisor dispatching to specialist agents. The
final-project spec explicitly forbids LangGraph, CrewAI, AutoGen, or similar
frameworks from providing the submitted project's main orchestration, agent
loop, or graph runtime — they may only be studied or benchmarked.

## Decision
Built `app/graph/engine.py` from scratch: a `Graph` of named nodes dispatching
on a field in the state object (`next_action`), a step-limit guard against
unintended cycles, a `TraceSink` hook emitting one span per transition, and a
generic `GraphContext` dependency bag. Both the single-agent graph
(`app/runtime.py`) and the multi-agent graph (`app/agents/graph.py`) are built
on this one engine, and `resume()` is the *same* `invoke()` call re-entered
from a suspended state, not a separate code path.

## Alternatives considered

**LangGraph.** Rejected outright by the assessment rules for the core
orchestrator. Independent of that rule, LangGraph's `StateGraph` would have
hidden the retry/backoff and approval-suspend logic behind framework
abstractions the team would need to defend without having written — the
project's learning objective (LO6/LO8/LO11: agentic workflow design, typed
state, explainable routing) is best met by owning the dispatch loop.

**A flat `if/elif` pipeline** (what the pre-existing `app/runtime.py::run()`
looked like before this work). Simpler to read for a single fixed sequence,
but it does not generalize: the multi-agent graph needs a *variable-length*
plan (the supervisor's step list) and a resumable suspend point mid-plan,
which a flat function cannot express without duplicating the whole dispatch
logic a second time. Confirmed painful in practice — the original `run()`
recomputed status strings and dispatch by hand for exactly three fixed steps.

**A generic library-free FSM (e.g. `transitions`).** Considered and rejected:
adds a dependency for something ~140 lines of engine code covers, and a
third-party FSM library still would not understand `AgentState`'s Pydantic
validators or the trace/retry integration this project needs.

## Consequences
- Every node is independently unit-testable as a plain function `(state, ctx) -> state`.
- `resume()` for both graphs is provably the same dispatch path as a fresh run — see `tests/test_graph_engine.py::test_invoke_starts_from_the_states_current_route_not_a_fixed_entry`.
- The trade-off: no built-in persistence, streaming-node, or distributed-execution story that a mature framework would offer. Not needed at this project's scale (single-process, in-memory approval store — already documented as a constraint in `app/approvals/store.py`), but would need to be built if the project scaled to multi-worker deployment.
