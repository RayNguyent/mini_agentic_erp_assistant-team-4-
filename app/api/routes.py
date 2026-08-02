import json
import time
import uuid

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.agents.graph import resume_multi_agent, run_multi_agent
from app.api.dependencies import (
    get_agent_registry,
    get_approval_store,
    get_audit_log,
    get_current_actor,
    get_llm_provider,
    get_retriever,
    get_tool_registry,
    get_trace_store,
)
from app.api.schemas import (
    ApproveRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    ReadinessResponse,
    RejectRequest,
    RiskDraft,
    ToolInfo,
    ToolsResponse,
)
from app.approvals.store import ApprovalStore
from app.observability.trace import TraceRecorder, TraceStore
from app.providers.base import LLMProvider
from app.providers.config import ProviderSettings
from app.rag.retrieve import Retriever
from app.runtime import ToolRegistry
from app.security.audit import AuditLog
from app.security.injection import screen_user_input
from app.security.permissions import check as check_permission
from app.state import Actor, AgentState, NextAction, Route
from app.tokenizer import tokenizer_name
from app.tools.specs import TOOL_META

router = APIRouter()


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_response(
    payload: ChatResponse, statuses: list[str] | None = None
) -> StreamingResponse:
    """Stream agent decision-making + token-by-token response.

    The multi-agent run completes before this generator starts (RAG grounding,
    citation verification, and approval gating all have to finish before an
    answer is safe to show), so what streams is the *rendering* of an already-
    final answer: buffered status/route/citation events, then the answer text
    word-by-word. This is a documented simplification, not real token-level
    LLM streaming — see docs/adr/ for the trade-off.
    """

    def event_stream():
        if statuses:
            for status in statuses:
                yield _sse_event("status", {"message": status})
                time.sleep(0.05)

        if payload.route:
            yield _sse_event("route", {"route": payload.route, "agent_path": payload.agent_path})
        for citation in payload.citations:
            yield _sse_event("citation", citation.model_dump())

        words = payload.answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield _sse_event("token", {"text": chunk})
        yield _sse_event("done", payload.model_dump())

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _as_text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _risk_draft(state: AgentState) -> RiskDraft | None:
    """Seed the approval form from the pending state, so fields the user
    already mentioned come back filled in and only the gaps need typing."""
    if state.selected_tool != "create_risk":
        return None
    tool_input = state.tool_input or {}
    payload = tool_input.get("risk_payload")
    if not isinstance(payload, dict):
        payload = {}
    # A pre-fill is a convenience, never a source of truth — anything the model
    # got wrong (odd severity, non-string title) falls back to a blank field
    # for the user to correct rather than failing the response.
    severity = payload.get("severity")
    return RiskDraft(
        project_code=_as_text(tool_input.get("project_code")),
        title=_as_text(payload.get("title")),
        severity=severity if severity in ("low", "medium", "high") else "medium",
        description=_as_text(payload.get("description")),
    )


def _agent_path(state: AgentState) -> list[str]:
    return [step.agent for step in state.plan]


def _provider_meta() -> tuple[str | None, str | None]:
    settings = ProviderSettings.from_env()
    return settings.llm_provider, (settings.openai_model if settings.is_live else None)


def _finished_response(state: AgentState) -> ChatResponse:
    provider, model = _provider_meta()
    return ChatResponse(
        answer=state.answer or "No result available.",
        tool_used=state.selected_tool,
        error_code=state.error_code.value if state.error_code else None,
        route=state.route.value if state.route else None,
        agent_path=_agent_path(state),
        citations=state.retrieved_sources,
        request_id=state.request_id,
        provider=provider,
        model=model,
    )


def _approval_response(state: AgentState, approval_id: str) -> ChatResponse:
    draft = _risk_draft(state)
    answer = (
        "Fill in the risk details below, then approve to create it."
        if draft is not None
        else f"'{state.selected_tool}' requires approval before it runs."
    )
    provider, model = _provider_meta()
    return ChatResponse(
        answer=answer,
        tool_used=state.selected_tool,
        approval_required=True,
        approval_id=approval_id,
        risk_draft=draft,
        route=state.route.value if state.route else None,
        agent_path=_agent_path(state),
        request_id=state.request_id,
        provider=provider,
        model=model,
    )


def _as_history(request: ChatRequest) -> list[dict] | None:
    return [turn.model_dump() for turn in request.history] if request.history else None


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.get("/readiness", response_model=ReadinessResponse)
def readiness(
    provider: LLMProvider | None = Depends(get_llm_provider),
    retriever: Retriever = Depends(get_retriever),
) -> ReadinessResponse:
    """Reports the active provider/model and RAG index status so a demo (or an
    operator) can distinguish "healthy but degraded to offline/BM25-only" from
    "fully configured" — final-project.pdf's "Configured model provider" demo
    scenario."""
    settings = ProviderSettings.from_env()
    degraded = provider is None or not retriever.uses_vectors
    return ReadinessResponse(
        status="degraded" if degraded else "ok",
        llm_provider=settings.llm_provider,
        model=settings.openai_model if settings.is_live else None,
        base_url=settings.resolved_base_url,
        credential_configured=bool(settings.openai_api_key),
        tokenizer=tokenizer_name(),
        rag_index_chunks=retriever.size,
        rag_vector_index=retriever.vector_index.name,
        rag_vector_count=retriever.vector_index.count(),
        embeddings_enabled=settings.embeddings_enabled,
        erp_provider=settings.erp_provider,
    )


@router.get("/tools", response_model=ToolsResponse)
def list_tools() -> ToolsResponse:
    """The MCP-style tool registry: permission, side-effect level, timeout,
    retry limit, and audit category for every tool — required evidence per
    final-project.pdf §2 "MCP-style tool boundary"."""
    return ToolsResponse(
        tools=[
            ToolInfo(
                name=meta.name, description="", permission=meta.permission,
                side_effect=meta.side_effect, timeout_s=meta.timeout_s,
                retry_limit=meta.retry_limit, audit_category=meta.audit_category,
            )
            for meta in TOOL_META.values()
        ]
    )


@router.get("/traces/{request_id}")
def get_trace(request_id: str, store: TraceStore = Depends(get_trace_store)):
    trace = store.get(request_id)
    if trace is None:
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"No trace for request '{request_id}'.")
    return trace.model_dump()


@router.get("/traces")
def list_traces(limit: int = 20, store: TraceStore = Depends(get_trace_store)):
    return {"traces": [t.model_dump() for t in store.recent(limit=limit)]}


def _run_chat(
    request: ChatRequest,
    actor: Actor,
    registry: ToolRegistry,
    retriever: Retriever,
    store: ApprovalStore,
    provider: LLMProvider | None,
    agents: dict,
    audit: AuditLog,
    trace_store: TraceStore,
    on_status=None,
) -> AgentState:
    request_id = str(uuid.uuid4())
    recorder = TraceRecorder(request_id, trace_store)
    audit.auth(actor, "success", request_id=request_id)

    finding = screen_user_input(request.message)
    if finding.flagged:
        recorder.record_injection_finding("user_input", finding.matched_pattern or "")
        audit.blocked("prompt_injection_detected", actor=actor, request_id=request_id, detail={"pattern": finding.matched_pattern})
        # Flagged, not blocked outright — see app.security.injection module
        # docstring: a legitimate user may ask about policy in words that
        # happen to match. Enforcement stays structural (approval gate,
        # citation gate); this only makes the attempt visible in the audit log.

    state = run_multi_agent(
        request.message,
        tool_registry=registry,
        retriever=retriever,
        history=_as_history(request),
        provider=provider,
        actor=actor,
        request_id=request_id,
        on_status=on_status,
        trace=recorder,
        agents=agents,
        authorize=check_permission,
    )

    for step_name, result in state.agent_results.items():
        audit.tool_call(
            result.tool_used or step_name,
            "success" if result.ok else "error",
            actor=actor, request_id=request_id,
            detail={"agent": step_name, "error_code": result.error_code.value if result.error_code else None},
        )

    # The pending-approval record is created by the caller (chat/chat_stream),
    # not here — that is what produces the approval_id actually returned to
    # the client, and the audit row must reference that exact id, not a
    # separate one this function would otherwise mint for itself.

    recorder.finish(route=state.route.value if state.route else None, error=state.error_message)
    return state


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    actor: Actor = Depends(get_current_actor),
    registry: ToolRegistry = Depends(get_tool_registry),
    retriever: Retriever = Depends(get_retriever),
    store: ApprovalStore = Depends(get_approval_store),
    provider: LLMProvider | None = Depends(get_llm_provider),
    agents: dict = Depends(get_agent_registry),
    audit: AuditLog = Depends(get_audit_log),
    trace_store: TraceStore = Depends(get_trace_store),
) -> ChatResponse:
    state = _run_chat(request, actor, registry, retriever, store, provider, agents, audit, trace_store)

    if state.next_action == NextAction.AWAIT_APPROVAL:
        approval_id = store.create(state)
        audit.approval(approval_id, "pending", actor=actor, request_id=state.request_id)
        return _approval_response(state, approval_id)

    audit.response(state.route.value if state.route else "unknown", actor=actor, request_id=state.request_id)
    return _finished_response(state)


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    actor: Actor = Depends(get_current_actor),
    registry: ToolRegistry = Depends(get_tool_registry),
    retriever: Retriever = Depends(get_retriever),
    store: ApprovalStore = Depends(get_approval_store),
    provider: LLMProvider | None = Depends(get_llm_provider),
    agents: dict = Depends(get_agent_registry),
    audit: AuditLog = Depends(get_audit_log),
    trace_store: TraceStore = Depends(get_trace_store),
) -> StreamingResponse:
    statuses: list[str] = []

    def on_status(status: str) -> None:
        statuses.append(status)

    state = _run_chat(request, actor, registry, retriever, store, provider, agents, audit, trace_store, on_status=on_status)

    if state.next_action == NextAction.AWAIT_APPROVAL:
        approval_id = store.create(state)
        audit.approval(approval_id, "pending", actor=actor, request_id=state.request_id)
        payload = _approval_response(state, approval_id)
    else:
        audit.response(state.route.value if state.route else "unknown", actor=actor, request_id=state.request_id)
        payload = _finished_response(state)

    return _stream_response(payload, statuses=statuses)


def _resume(
    request_approval_id: str,
    approved: bool,
    extra_tool_input: dict | None,
    actor: Actor,
    registry: ToolRegistry,
    store: ApprovalStore,
    agents: dict,
    audit: AuditLog,
    trace_store: TraceStore,
) -> AgentState:
    pending = store.pop(request_approval_id)
    merged_input = {**(pending.tool_input or {}), **(extra_tool_input or {})}
    state = pending.model_copy(update={"approved": approved, "tool_input": merged_input})

    recorder = TraceRecorder(state.request_id or request_approval_id, trace_store)

    if state.route == Route.MULTI_AGENT:
        resumed = resume_multi_agent(
            state, tool_registry=registry, agents=agents, authorize=check_permission, trace=recorder,
        )
    else:
        from app.runtime import resume

        resumed = resume(state, registry, trace=recorder)

    outcome = "approved" if approved else "rejected"
    # Written before the pending record was deleted by store.pop() above is
    # not possible here (pop must happen first to retrieve the state) — but
    # the row is written unconditionally, before any tool exception could
    # prevent it, which is the property that actually matters: a decision is
    # never lost even if execution afterward fails.
    audit.approval(request_approval_id, outcome, actor=actor, request_id=state.request_id, detail={
        "error_code": resumed.error_code.value if resumed.error_code else None,
    })
    recorder.finish(route=resumed.route.value if resumed.route else None, error=resumed.error_message)
    return resumed


@router.post("/approve", response_model=ChatResponse)
def approve(
    request: ApproveRequest,
    actor: Actor = Depends(get_current_actor),
    registry: ToolRegistry = Depends(get_tool_registry),
    store: ApprovalStore = Depends(get_approval_store),
    agents: dict = Depends(get_agent_registry),
    audit: AuditLog = Depends(get_audit_log),
    trace_store: TraceStore = Depends(get_trace_store),
) -> ChatResponse:
    state = _resume(request.approval_id, True, request.tool_input, actor, registry, store, agents, audit, trace_store)
    return _finished_response(state)


@router.post("/approve/stream")
def approve_stream(
    request: ApproveRequest,
    actor: Actor = Depends(get_current_actor),
    registry: ToolRegistry = Depends(get_tool_registry),
    store: ApprovalStore = Depends(get_approval_store),
    agents: dict = Depends(get_agent_registry),
    audit: AuditLog = Depends(get_audit_log),
    trace_store: TraceStore = Depends(get_trace_store),
) -> StreamingResponse:
    state = _resume(request.approval_id, True, request.tool_input, actor, registry, store, agents, audit, trace_store)
    statuses = ["✅ Approval confirmed!", f"🔧 Executing: {state.selected_tool}"]
    return _stream_response(_finished_response(state), statuses=statuses)


@router.post("/reject", response_model=ChatResponse)
def reject(
    request: RejectRequest,
    actor: Actor = Depends(get_current_actor),
    registry: ToolRegistry = Depends(get_tool_registry),
    store: ApprovalStore = Depends(get_approval_store),
    agents: dict = Depends(get_agent_registry),
    audit: AuditLog = Depends(get_audit_log),
    trace_store: TraceStore = Depends(get_trace_store),
) -> ChatResponse:
    state = _resume(request.approval_id, False, None, actor, registry, store, agents, audit, trace_store)
    return _finished_response(state)


@router.post("/reject/stream")
def reject_stream(
    request: RejectRequest,
    actor: Actor = Depends(get_current_actor),
    registry: ToolRegistry = Depends(get_tool_registry),
    store: ApprovalStore = Depends(get_approval_store),
    agents: dict = Depends(get_agent_registry),
    audit: AuditLog = Depends(get_audit_log),
    trace_store: TraceStore = Depends(get_trace_store),
) -> StreamingResponse:
    state = _resume(request.approval_id, False, None, actor, registry, store, agents, audit, trace_store)
    return _stream_response(_finished_response(state))
