import pytest
from pydantic import BaseModel

from app.errors import NotFoundError, ToolTimeoutError
from app.graph.engine import (
    Graph,
    GraphContext,
    NodeNotFoundError,
    NodeSpan,
    StepLimitExceeded,
    evolve,
)
from app.graph.retry import RetryPolicy, retry_call


class Toy(BaseModel):
    next_action: str
    trail: list[str] = []
    counter: int = 0


def _graph() -> Graph:
    def a(state, ctx):
        return evolve(state, trail=[*state.trail, "a"], next_action="b")

    def b(state, ctx):
        return evolve(state, trail=[*state.trail, "b"], next_action="done")

    return Graph("toy").add_node("a", a).add_node("b", b).add_terminal("done")


# --- dispatch ---------------------------------------------------------------


def test_invoke_walks_nodes_until_a_terminal():
    state = _graph().invoke(Toy(next_action="a"), GraphContext())
    assert state.trail == ["a", "b"]
    assert state.next_action == "done"


def test_invoke_starts_from_the_states_current_route_not_a_fixed_entry():
    # This is what makes resume() the same code path as a fresh run.
    state = _graph().invoke(Toy(next_action="b"), GraphContext())
    assert state.trail == ["b"]


def test_invoke_returns_immediately_when_already_terminal():
    state = _graph().invoke(Toy(next_action="done"), GraphContext())
    assert state.trail == []


def test_unknown_node_raises_with_the_known_names():
    with pytest.raises(NodeNotFoundError, match="no node 'nope'"):
        _graph().invoke(Toy(next_action="nope"), GraphContext())


def test_cycle_is_bounded_by_the_step_limit():
    def loop(state, ctx):
        return evolve(state, counter=state.counter + 1, next_action="loop")

    graph = Graph("cyclic", max_steps=5).add_node("loop", loop).add_terminal("done")
    with pytest.raises(StepLimitExceeded, match="exceeded 5 steps"):
        graph.invoke(Toy(next_action="loop"), GraphContext())


def test_a_bounded_cycle_is_allowed_to_finish():
    def loop(state, ctx):
        nxt = "done" if state.counter >= 2 else "loop"
        return evolve(state, counter=state.counter + 1, next_action=nxt)

    graph = Graph("bounded", max_steps=10).add_node("loop", loop).add_terminal("done")
    assert graph.invoke(Toy(next_action="loop"), GraphContext()).counter == 3


# --- evolve re-runs validators ----------------------------------------------


class Guarded(BaseModel):
    next_action: str
    value: int = 0

    def model_post_init(self, _ctx) -> None:
        if self.value < 0:
            raise ValueError("value must not be negative")


def test_evolve_rebuilds_through_the_constructor_so_validators_run():
    with pytest.raises(ValueError, match="must not be negative"):
        evolve(Guarded(next_action="a"), value=-1)


# --- tracing ----------------------------------------------------------------


class RecordingSink:
    def __init__(self):
        self.spans: list[NodeSpan] = []

    def record_node(self, span: NodeSpan) -> None:
        self.spans.append(span)


def test_every_node_transition_emits_one_span():
    sink = RecordingSink()
    _graph().invoke(Toy(next_action="a"), GraphContext(trace=sink))

    assert [(s.node, s.next_node) for s in sink.spans] == [("a", "b"), ("b", "done")]
    assert [s.step for s in sink.spans] == [1, 2]
    assert all(s.error is None for s in sink.spans)


def test_a_raising_node_is_traced_before_the_exception_propagates():
    sink = RecordingSink()

    def boom(state, ctx):
        raise RuntimeError("kaboom")

    graph = Graph("boom").add_node("a", boom).add_terminal("done")
    with pytest.raises(RuntimeError, match="kaboom"):
        graph.invoke(Toy(next_action="a"), GraphContext(trace=sink))

    assert sink.spans[0].error == "RuntimeError: kaboom"


# --- context ----------------------------------------------------------------


def test_require_raises_for_a_missing_dependency():
    with pytest.raises(Exception, match="missing required dependency 'tool_registry'"):
        GraphContext().require("tool_registry")


def test_child_overrides_without_mutating_the_parent():
    parent = GraphContext(tool_registry="full", message="hi")
    child = parent.child(tool_registry="narrowed")
    assert child.get("tool_registry") == "narrowed"
    assert child.get("message") == "hi"
    assert parent.get("tool_registry") == "full"


def test_emit_is_a_noop_without_a_callback():
    GraphContext().emit("nothing listening")


# --- retry ------------------------------------------------------------------


def _retryable(exc: Exception) -> bool:
    return isinstance(exc, ToolTimeoutError)


def test_retry_call_returns_the_attempt_count_used():
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ToolTimeoutError("transient")
        return "ok"

    result, retries = retry_call(
        flaky,
        policy=RetryPolicy(max_retries=2, jitter=False),
        is_retryable=_retryable,
        sleep=lambda _: None,
    )
    assert (result, retries, len(calls)) == ("ok", 2, 3)


def test_retry_call_does_not_retry_a_deterministic_failure():
    calls = []

    def not_found():
        calls.append(1)
        raise NotFoundError("PRJ-000")

    with pytest.raises(NotFoundError):
        retry_call(
            not_found,
            policy=RetryPolicy(max_retries=5, jitter=False),
            is_retryable=_retryable,
            sleep=lambda _: None,
        )
    assert len(calls) == 1


def test_retry_call_gives_up_after_the_budget_and_reraises():
    calls = []
    delays = []

    def always_fails():
        calls.append(1)
        raise ToolTimeoutError("still down")

    with pytest.raises(ToolTimeoutError):
        retry_call(
            always_fails,
            policy=RetryPolicy(max_retries=2, jitter=False),
            is_retryable=_retryable,
            sleep=delays.append,
        )
    assert len(calls) == 3  # initial + 2 retries
    assert delays == [0.05, 0.1]  # exponential


def test_backoff_is_jittered_so_retries_do_not_synchronise():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=100.0)
    samples = {policy.delay_for(3) for _ in range(50)}
    assert len(samples) > 1  # not a constant
    assert all(0 <= s <= 8.0 for s in samples)  # within base * 2**3


def test_backoff_is_capped():
    policy = RetryPolicy(base_delay_s=1.0, max_delay_s=2.0, jitter=False)
    assert policy.delay_for(10) == 2.0
