"""Trace capture: one request_id, many spans, exported for replay.

Implements the TraceSink protocol app.graph.engine.Graph expects
(`record_node`), plus richer span types (retrieval, agent hop, LLM call, tool
call) that the RAG loop and the multi-agent graph record directly via
`ctx.trace`. A trace answers "what happened for this request" without reading
logs: prompts, retrieval, tool calls, latency, errors, and the final route,
all under one request_id — which is exactly the replay evidence the
evaluation runner and the golden-case failure analysis both read.
"""

import json
import threading
import time
from pathlib import Path

from pydantic import BaseModel, Field

from app.graph.engine import NodeSpan

TRACE_LOG_PATH = Path("data/traces.jsonl")


class Span(BaseModel):
    """One recorded event within a trace. `kind` distinguishes node
    transitions from the richer domain events (retrieval, llm_call, ...)."""

    kind: str
    name: str
    started_at: float
    elapsed_ms: float
    attributes: dict = Field(default_factory=dict)
    error: str | None = None


class Trace(BaseModel):
    request_id: str
    started_at: float
    spans: list[Span] = Field(default_factory=list)
    route: str | None = None
    final_error: str | None = None

    @property
    def total_ms(self) -> float:
        return sum(s.elapsed_ms for s in self.spans)


class TraceRecorder:
    """Accumulates spans for one request_id, then flushes to the append-only
    store on `finish()`. A new instance per request keeps concurrent requests'
    traces from interleaving."""

    def __init__(self, request_id: str, store: "TraceStore | None" = None) -> None:
        self.request_id = request_id
        self._store = store
        self._trace = Trace(request_id=request_id, started_at=time.time())

    # --- TraceSink protocol (app.graph.engine.Graph calls this) -----------

    def record_node(self, span: NodeSpan) -> None:
        self._trace.spans.append(
            Span(
                kind="node",
                name=span.node,
                started_at=span.started_at,
                elapsed_ms=span.elapsed_ms,
                attributes={**span.attributes, "next_node": span.next_node, "step": span.step},
                error=span.error,
            )
        )

    # --- richer domain spans -------------------------------------------

    def record_retrieval(self, diagnostics: dict, *, elapsed_ms: float = 0.0) -> None:
        self._trace.spans.append(
            Span(kind="retrieval", name="retrieve", started_at=time.time(), elapsed_ms=elapsed_ms, attributes=diagnostics)
        )

    def record_agent(self, agent: str, *, ok: bool, elapsed_ms: float, attributes: dict | None = None) -> None:
        self._trace.spans.append(
            Span(
                kind="agent", name=agent, started_at=time.time(), elapsed_ms=elapsed_ms,
                attributes=attributes or {}, error=None if ok else "agent reported failure",
            )
        )

    def record_llm_call(self, model: str, *, tokens_in: int = 0, tokens_out: int = 0, elapsed_ms: float = 0.0, cost_usd: float | None = None) -> None:
        self._trace.spans.append(
            Span(
                kind="llm_call", name=model, started_at=time.time(), elapsed_ms=elapsed_ms,
                attributes={"tokens_in": tokens_in, "tokens_out": tokens_out, "cost_usd": cost_usd},
            )
        )

    def record_tool_call(self, tool_name: str, *, ok: bool, elapsed_ms: float, retries: int = 0) -> None:
        self._trace.spans.append(
            Span(
                kind="tool_call", name=tool_name, started_at=time.time(), elapsed_ms=elapsed_ms,
                attributes={"retries": retries}, error=None if ok else "tool call failed",
            )
        )

    def record_context_build(self, log: dict) -> None:
        """What was admitted into the LLM prompt under the token budget, and
        what was excluded and why (app.context.builder.ContextBuildLog,
        serialized). Recorded once per generation call — the RAG loop's
        `_generate_node` is the current caller."""
        self._trace.spans.append(
            Span(
                kind="context_build", name="build_context", started_at=time.time(),
                elapsed_ms=0.0, attributes=log,
            )
        )

    def record_injection_finding(self, source: str, matched_pattern: str) -> None:
        self._trace.spans.append(
            Span(kind="injection_finding", name=source, started_at=time.time(), elapsed_ms=0.0,
                 attributes={"matched_pattern": matched_pattern})
        )

    def record_memory_decision(self, decision, *, agent: str) -> None:
        """One MemoryDecision (SAVE/UPDATE/IGNORE/EXPIRE/REJECT), as returned
        by app.memory.store.MemoryStore.propose(). Recorded regardless of
        action — a REJECT is exactly as important to see in a trace as a
        SAVE."""
        self._trace.spans.append(
            Span(
                kind="memory_decision", name=decision.action.value, started_at=time.time(), elapsed_ms=0.0,
                attributes={
                    "agent": agent,
                    "reason": decision.reason,
                    "source": decision.candidate.source,
                    "subject": decision.candidate.subject,
                    "target_memory_id": decision.target_memory_id,
                },
            )
        )

    # --- lifecycle -----------------------------------------------------

    def finish(self, *, route: str | None = None, error: str | None = None) -> Trace:
        self._trace.route = route
        self._trace.final_error = error
        if self._store is not None:
            self._store.save(self._trace)
        return self._trace

    @property
    def trace(self) -> Trace:
        return self._trace


class TraceStore:
    """Append-only JSONL sink plus an in-memory index for `GET /traces`."""

    def __init__(self, path: Path | str = TRACE_LOG_PATH) -> None:
        self._path = Path(path)
        self._lock = threading.Lock()
        self._traces: dict[str, Trace] = {}

    def save(self, trace: Trace) -> None:
        with self._lock:
            self._traces[trace.request_id] = trace
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as handle:
                handle.write(trace.model_dump_json() + "\n")

    def get(self, request_id: str) -> Trace | None:
        return self._traces.get(request_id)

    def recent(self, limit: int = 20) -> list[Trace]:
        return list(self._traces.values())[-limit:]


def build_default_trace_store() -> TraceStore:
    return TraceStore()
