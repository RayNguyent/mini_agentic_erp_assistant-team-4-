"""The multi-agent graph: supervise -> run_agents -> synthesize.

    supervise ──plan──> run_agents ──(all steps done)──> synthesize ──> done
                             │
                             ├─ a step is create_risk with no decision yet
                             └─> AWAIT_APPROVAL  (top-level state suspends)
                                       │
                            /approve or /reject sets `approved`
                                       │
                                  resume() re-enters run_agents,
                                  skips already-completed steps
                                  (kept in state.agent_results),
                                  executes the pending write, continues.

Approval reuses the exact mechanism the single-agent graph uses —
`execute_write_tool` inside RiskWriter, the same ApprovalStore, the same
AgentState.tool_output/approval validators — so a write's safety guarantee
does not fork into a second implementation for the multi-agent path.
"""

from app.agents.base import SpecialistAgent
from app.agents.doc_researcher import DocResearcher
from app.agents.erp_analyst import ERPAnalyst
from app.agents.risk_writer import RiskWriter
from app.agents.supervisor import plan as plan_steps
from app.graph.retry import RetryPolicy
from app.runtime import DEFAULT_RETRY_POLICY
from app.agents.synthesizer import synthesize
from app.errors import ErrorCode
from app.graph.engine import Graph, GraphContext, evolve
from app.state import AgentResult, AgentState, NextAction, Route


def default_agent_registry() -> dict[str, SpecialistAgent]:
    return {
        "erp_analyst": ERPAnalyst(),
        "doc_researcher": DocResearcher(),
        "risk_writer": RiskWriter(),
    }


def _supervise_node(state: AgentState, ctx: GraphContext) -> AgentState:
    ctx.emit("🧭 Planning specialist steps...")
    steps = plan_steps(ctx.require("message"), ctx.get("history"), ctx.get("provider"))
    names = ", ".join(f"{s.agent}" for s in steps)
    ctx.emit(f"✓ Plan: {names}")
    return evolve(state, plan=steps, route=Route.MULTI_AGENT, next_action=NextAction.RUN_AGENTS)


def _run_agents_node(state: AgentState, ctx: GraphContext) -> AgentState:
    agents: dict[str, SpecialistAgent] = ctx.get("agents") or default_agent_registry()
    results = dict(state.agent_results)  # steps already completed survive a resume

    for step in state.plan:
        if step.agent in results:
            continue  # completed before a prior suspend

        agent = agents.get(step.agent)
        if agent is None:
            results[step.agent] = AgentResult(
                agent=step.agent,
                ok=False,
                error_code=ErrorCode.NOT_FOUND,
                error_message=f"no specialist registered for '{step.agent}'",
            )
            continue

        ctx.emit(f"🤖 {step.agent}: {step.rationale or 'running'}")
        result = agent.run(step, state, ctx)

        if result.awaiting_approval:
            # Suspend the whole multi-agent run, not just this step. The
            # pending tool_input mirrors exactly what the single-agent graph
            # puts on state before AWAIT_APPROVAL, so the same ApprovalStore /
            # /approve /reject flow works unchanged.
            return evolve(
                state,
                agent_results=results,
                selected_tool="create_risk",
                tool_input=step.inputs.get("tool_input") or {},
                approval_required=True,
                risk_level="high",
                next_action=NextAction.AWAIT_APPROVAL,
                status=f"'{step.agent}' requires approval before it runs.",
            )

        results[step.agent] = result
        ctx.emit(f"✓ {step.agent} done ({result.elapsed_ms:.0f}ms)")

    return evolve(state, agent_results=results, next_action=NextAction.SYNTHESIZE)


def _synthesize_node(state: AgentState, ctx: GraphContext) -> AgentState:
    ordered = [state.agent_results[step.agent] for step in state.plan if step.agent in state.agent_results]
    answer, citations, all_failed = synthesize(ordered)

    # A refusal is only unambiguous for a single doc_researcher step: it
    # returned ok=True (it did its job) with zero citations, meaning the
    # corpus genuinely had nothing. A multi-specialist run is never marked as
    # a refusal that way — a partial answer from the other agents is still
    # worth showing even if the document arm came up empty.
    refused = (
        len(ordered) == 1
        and ordered[0].agent == "doc_researcher"
        and ordered[0].ok
        and not ordered[0].citations
    )

    # error_code: for a single specialist, surface its error_code verbatim —
    # this covers APPROVAL_REJECTED, which sets ok=True (the gate worked as
    # intended) but must still reach the client as an error_code, matching
    # the single-agent API's existing contract. For multiple specialists,
    # only report a top-level error_code when every one of them failed.
    if len(ordered) == 1:
        error_code = ordered[0].error_code
        error_message = ordered[0].error_message
    elif all_failed and ordered:
        error_code = next((r.error_code for r in ordered if r.error_code), ErrorCode.UNKNOWN)
        error_message = ordered[0].error_message
    else:
        error_code = None
        error_message = None

    # tool_used: only unambiguous for a single specialist's tool-backed result.
    selected_tool = ordered[0].tool_used if len(ordered) == 1 else None

    return evolve(
        state,
        answer=answer,
        retrieved_sources=citations,
        refused=refused,
        selected_tool=selected_tool,
        error_code=error_code if not refused else None,
        error_message=error_message if not refused else None,
        next_action=NextAction.DONE,
        status="Done!",
    )


def build_multi_agent_graph() -> Graph:
    return (
        Graph("multi_agent")
        .add_node(NextAction.SUPERVISE, _supervise_node)
        .add_node(NextAction.RUN_AGENTS, _run_agents_node)
        .add_node(NextAction.SYNTHESIZE, _synthesize_node)
        .add_terminal(NextAction.DONE, NextAction.AWAIT_APPROVAL)
    )


MULTI_AGENT_GRAPH = build_multi_agent_graph()


def run_multi_agent(
    message: str,
    *,
    tool_registry,
    retriever,
    history: list[dict] | None = None,
    provider=None,
    actor=None,
    request_id: str | None = None,
    on_status=None,
    trace=None,
    agents: dict[str, SpecialistAgent] | None = None,
    authorize=None,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    memory_store=None,
    audit_log=None,
) -> AgentState:
    ctx = GraphContext(
        trace=trace,
        message=message,
        history=history,
        provider=provider,
        tool_registry=tool_registry,
        retriever=retriever,
        on_status=on_status,
        agents=agents,
        authorize=authorize,
        retry_policy=retry_policy,
        memory_store=memory_store,
        audit_log=audit_log,
    )
    initial = AgentState(
        intent="multi_agent",
        next_action=NextAction.SUPERVISE,
        request_id=request_id,
        actor=actor,
    )
    return MULTI_AGENT_GRAPH.invoke(initial, ctx)


def resume_multi_agent(
    state: AgentState,
    *,
    tool_registry,
    retriever=None,
    on_status=None,
    trace=None,
    agents: dict[str, SpecialistAgent] | None = None,
    authorize=None,
    retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    memory_store=None,
    audit_log=None,
) -> AgentState:
    """Continue a multi-agent run after /approve or /reject has set `approved`."""
    if state.next_action != NextAction.AWAIT_APPROVAL:
        raise ValueError(
            f"resume_multi_agent() called on a state that is not awaiting approval: {state.next_action}"
        )
    ctx = GraphContext(
        trace=trace,
        message="",
        tool_registry=tool_registry,
        retriever=retriever,
        on_status=on_status,
        agents=agents,
        authorize=authorize,
        retry_policy=retry_policy,
        memory_store=memory_store,
        audit_log=audit_log,
    )
    return MULTI_AGENT_GRAPH.invoke(evolve(state, next_action=NextAction.RUN_AGENTS), ctx)
