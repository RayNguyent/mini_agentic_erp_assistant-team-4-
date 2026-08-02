"""Trace capture: node spans wired through the graph engine, richer domain
spans, and JSONL persistence."""

import json

from app.graph.engine import GraphContext
from app.observability.trace import TraceRecorder, TraceStore
from app.runtime import GRAPH, default_classify
from app.state import AgentState, NextAction
from app.tools.registry import build_default_registry


def test_recorder_satisfies_the_graph_engines_trace_sink_protocol():
    recorder = TraceRecorder("req-1")
    registry = build_default_registry()
    ctx = GraphContext(
        trace=recorder,
        message="What's the status of PRJ-001?",
        tool_registry=registry,
        classify=default_classify,
    )
    initial = AgentState(intent="unknown", next_action=NextAction.PARSE_INTENT)

    GRAPH.invoke(initial, ctx)

    node_spans = [s for s in recorder.trace.spans if s.kind == "node"]
    assert len(node_spans) >= 3  # parse_intent -> route_decision -> execute_read_tool -> format_response
    assert node_spans[0].name == "parse_intent"


def test_a_failing_node_is_recorded_with_its_error():
    from pydantic import BaseModel

    from app.graph.engine import Graph

    recorder = TraceRecorder("req-err")

    def boom(state, ctx):
        raise RuntimeError("kaboom")

    class ToyState(BaseModel):
        next_action: str

    graph = Graph("boom").add_node("a", boom).add_terminal("done")

    try:
        graph.invoke(ToyState(next_action="a"), GraphContext(trace=recorder))
    except RuntimeError:
        pass

    assert recorder.trace.spans[0].error == "RuntimeError: kaboom"


def test_domain_spans_are_recorded_with_their_kind():
    recorder = TraceRecorder("req-2")
    recorder.record_retrieval({"bm25_hits": 2, "vector_hits": 0}, elapsed_ms=5.0)
    recorder.record_agent("doc_researcher", ok=True, elapsed_ms=12.0)
    recorder.record_tool_call("get_project_status", ok=True, elapsed_ms=3.0, retries=1)
    recorder.record_llm_call("gpt-4o-mini", tokens_in=100, tokens_out=20, elapsed_ms=200.0)
    recorder.record_injection_finding("document", "ignore instructions")

    kinds = [s.kind for s in recorder.trace.spans]
    assert kinds == ["retrieval", "agent", "tool_call", "llm_call", "injection_finding"]
    assert recorder.trace.spans[2].attributes["retries"] == 1


def test_finish_sets_route_and_persists_to_the_store(tmp_path):
    store = TraceStore(tmp_path / "traces.jsonl")
    recorder = TraceRecorder("req-3", store)
    recorder.record_tool_call("list_risks", ok=True, elapsed_ms=1.0)

    trace = recorder.finish(route="tool")

    assert trace.route == "tool"
    assert store.get("req-3") is trace
    lines = (tmp_path / "traces.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(lines[0])["request_id"] == "req-3"


def test_store_recent_returns_the_last_n_traces(tmp_path):
    store = TraceStore(tmp_path / "traces.jsonl")
    for i in range(5):
        TraceRecorder(f"req-{i}", store).finish()
    ids = [t.request_id for t in store.recent(limit=2)]
    assert ids == ["req-3", "req-4"]


def test_total_ms_sums_every_recorded_span():
    recorder = TraceRecorder("req-4")
    recorder.record_tool_call("a", ok=True, elapsed_ms=10.0)
    recorder.record_tool_call("b", ok=True, elapsed_ms=15.0)
    assert recorder.trace.total_ms == 25.0


def test_get_returns_none_for_an_unknown_request_id(tmp_path):
    store = TraceStore(tmp_path / "traces.jsonl")
    assert store.get("nope") is None
