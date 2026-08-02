"""A small, dependency-free graph execution engine.

Nodes are plain callables ``(state, ctx) -> state``. Control flow lives in the
state itself: every node sets the routing field (``next_action``) and the
driver dispatches on it. There is no separate edge list to drift out of sync
with the node bodies, and a node stays independently callable in a unit test.

The engine owns exactly four things the runtime should not re-implement per
graph: dispatch, a step limit (cycle protection), terminal/suspend detection,
and one trace span per node transition.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Protocol, TypeVar

from pydantic import BaseModel

StateT = TypeVar("StateT", bound=BaseModel)


class GraphError(RuntimeError):
    """Base class for failures of the engine itself, not of a node's work."""


class NodeNotFoundError(GraphError):
    """The state routed to a node name the graph does not define."""


class StepLimitExceeded(GraphError):
    """The graph exceeded ``max_steps`` — almost always an unintended cycle."""


def evolve(state: StateT, **updates: Any) -> StateT:
    """Build the next state through the constructor (not ``model_copy``) so the
    model's validators actually run on every transition.

    This is the single most important invariant in the runtime: an illegal
    state (e.g. a write tool's output without a recorded approval) raises at
    the moment of the transition that would create it, not later at a boundary.
    """
    data = state.model_dump()
    data.update(updates)
    return type(state)(**data)


def _name_of(value: Any) -> str:
    return value.value if isinstance(value, Enum) else str(value)


@dataclass(slots=True)
class NodeSpan:
    """One node execution, as recorded for the trace package."""

    node: str
    step: int
    started_at: float
    elapsed_ms: float
    next_node: str
    error: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)


class TraceSink(Protocol):
    """Structural contract the observability layer satisfies. Kept as a
    Protocol so the engine never imports the concrete recorder."""

    def record_node(self, span: NodeSpan) -> None: ...


class GraphContext:
    """Per-request dependency bag handed to every node.

    Nodes receive dependencies here rather than through closures so the same
    node function is reusable across graphs (main runtime, specialist
    subgraphs, evaluation harness) with different wiring.
    """

    __slots__ = ("_deps", "trace")

    def __init__(self, *, trace: TraceSink | None = None, **deps: Any) -> None:
        self._deps = deps
        self.trace = trace

    def get(self, key: str, default: Any = None) -> Any:
        return self._deps.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self._deps or self._deps[key] is None:
            raise GraphError(f"GraphContext is missing required dependency '{key}'")
        return self._deps[key]

    def emit(self, status: str) -> None:
        """Push a user-visible progress line, if the caller wired one up."""
        callback = self._deps.get("on_status")
        if callback is not None:
            callback(status)

    def child(self, **overrides: Any) -> "GraphContext":
        """A copy with some dependencies replaced — used to hand a specialist
        agent a narrowed tool registry without mutating the parent context."""
        merged = {**self._deps, **overrides}
        return GraphContext(trace=self.trace, **merged)


class Graph:
    """A named set of nodes plus the terminal/suspend states that stop the loop."""

    def __init__(
        self,
        name: str,
        *,
        route_field: str = "next_action",
        max_steps: int = 25,
    ) -> None:
        self.name = name
        self._route_field = route_field
        self._max_steps = max_steps
        self._nodes: dict[str, Callable[[Any, GraphContext], Any]] = {}
        self._terminals: set[str] = set()

    def add_node(self, name: Any, fn: Callable[[Any, GraphContext], Any]) -> "Graph":
        self._nodes[_name_of(name)] = fn
        return self

    def add_terminal(self, *names: Any) -> "Graph":
        """Mark states the driver stops on. Covers both true terminals (``done``)
        and suspend points (``await_approval``) — from the engine's point of
        view a suspended run is just a run that stopped and can be re-entered."""
        self._terminals.update(_name_of(n) for n in names)
        return self

    @property
    def node_names(self) -> list[str]:
        return sorted(self._nodes)

    def _route(self, state: Any) -> str:
        return _name_of(getattr(state, self._route_field))

    def invoke(self, state: StateT, ctx: GraphContext) -> StateT:
        """Drive `state` until it reaches a terminal/suspend node.

        Dispatch starts from the state's current routing field, which is what
        makes `resume` the same code path as a fresh run.
        """
        step = 0
        current = self._route(state)

        while current not in self._terminals:
            node = self._nodes.get(current)
            if node is None:
                raise NodeNotFoundError(
                    f"graph '{self.name}' has no node '{current}' "
                    f"(known: {', '.join(self.node_names)})"
                )

            step += 1
            if step > self._max_steps:
                raise StepLimitExceeded(
                    f"graph '{self.name}' exceeded {self._max_steps} steps at node "
                    f"'{current}' — check for a cycle in next_action transitions"
                )

            started = time.perf_counter()
            error: str | None = None
            try:
                state = node(state, ctx)
            except GraphError:
                raise
            except Exception as exc:  # noqa: BLE001 - recorded, then re-raised
                error = f"{type(exc).__name__}: {exc}"
                self._record(ctx, current, step, started, current, error)
                raise

            next_node = self._route(state)
            self._record(ctx, current, step, started, next_node, error)
            current = next_node

        return state

    def _record(
        self,
        ctx: GraphContext,
        node: str,
        step: int,
        started: float,
        next_node: str,
        error: str | None,
    ) -> None:
        if ctx.trace is None:
            return
        ctx.trace.record_node(
            NodeSpan(
                node=node,
                step=step,
                started_at=started,
                elapsed_ms=(time.perf_counter() - started) * 1000,
                next_node=next_node,
                error=error,
                attributes={"graph": self.name},
            )
        )
