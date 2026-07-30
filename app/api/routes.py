import json

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from app.api.dependencies import get_approval_store, get_intent_classifier, get_tool_registry
from app.api.schemas import (
    ApproveRequest,
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RejectRequest,
)
from app.approvals.store import ApprovalStore
from app.runtime import IntentClassifier, ToolRegistry, resume, run
from app.state import AgentState, NextAction

router = APIRouter()


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def _stream_response(payload: ChatResponse) -> StreamingResponse:
    """Fake token-by-token streaming over an already-computed ChatResponse.

    The runtime resolves an answer synchronously (no LLM generates the final
    text token-by-token), so this chunks the finished string client-side of
    the model call, purely to drive a ChatGPT-style typing UI.
    """

    def event_stream():
        words = payload.answer.split(" ")
        for i, word in enumerate(words):
            chunk = word if i == len(words) - 1 else word + " "
            yield _sse_event("token", {"text": chunk})
        yield _sse_event("done", payload.model_dump())

    return StreamingResponse(event_stream(), media_type="text/event-stream")


def _finished_response(state: AgentState) -> ChatResponse:
    return ChatResponse(
        answer=state.answer or "No result available.",
        tool_used=state.selected_tool,
        error_code=state.error_code.value if state.error_code else None,
    )


def _as_history(request: ChatRequest) -> list[dict] | None:
    return [turn.model_dump() for turn in request.history] if request.history else None


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    registry: ToolRegistry = Depends(get_tool_registry),
    classify: IntentClassifier = Depends(get_intent_classifier),
    store: ApprovalStore = Depends(get_approval_store),
) -> ChatResponse:
    state = run(request.message, registry, classify, history=_as_history(request))

    if state.next_action == NextAction.AWAIT_APPROVAL:
        approval_id = store.create(state)
        return ChatResponse(
            answer=f"'{state.selected_tool}' requires approval before it runs.",
            tool_used=state.selected_tool,
            approval_required=True,
            approval_id=approval_id,
        )

    return _finished_response(state)


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    registry: ToolRegistry = Depends(get_tool_registry),
    classify: IntentClassifier = Depends(get_intent_classifier),
    store: ApprovalStore = Depends(get_approval_store),
) -> StreamingResponse:
    state = run(request.message, registry, classify, history=_as_history(request))

    if state.next_action == NextAction.AWAIT_APPROVAL:
        approval_id = store.create(state)
        payload = ChatResponse(
            answer=f"'{state.selected_tool}' requires approval before it runs.",
            tool_used=state.selected_tool,
            approval_required=True,
            approval_id=approval_id,
        )
    else:
        payload = _finished_response(state)

    return _stream_response(payload)


@router.post("/approve", response_model=ChatResponse)
def approve(
    request: ApproveRequest,
    registry: ToolRegistry = Depends(get_tool_registry),
    store: ApprovalStore = Depends(get_approval_store),
) -> ChatResponse:
    pending = store.pop(request.approval_id)
    merged_input = {**(pending.tool_input or {}), **(request.tool_input or {})}
    state = pending.model_copy(update={"approved": True, "tool_input": merged_input})
    return _finished_response(resume(state, registry))


@router.post("/approve/stream")
def approve_stream(
    request: ApproveRequest,
    registry: ToolRegistry = Depends(get_tool_registry),
    store: ApprovalStore = Depends(get_approval_store),
) -> StreamingResponse:
    pending = store.pop(request.approval_id)
    merged_input = {**(pending.tool_input or {}), **(request.tool_input or {})}
    state = pending.model_copy(update={"approved": True, "tool_input": merged_input})
    return _stream_response(_finished_response(resume(state, registry)))


@router.post("/reject", response_model=ChatResponse)
def reject(
    request: RejectRequest,
    registry: ToolRegistry = Depends(get_tool_registry),
    store: ApprovalStore = Depends(get_approval_store),
) -> ChatResponse:
    pending = store.pop(request.approval_id)
    state = pending.model_copy(update={"approved": False})
    return _finished_response(resume(state, registry))


@router.post("/reject/stream")
def reject_stream(
    request: RejectRequest,
    registry: ToolRegistry = Depends(get_tool_registry),
    store: ApprovalStore = Depends(get_approval_store),
) -> StreamingResponse:
    pending = store.pop(request.approval_id)
    state = pending.model_copy(update={"approved": False})
    return _stream_response(_finished_response(resume(state, registry)))
