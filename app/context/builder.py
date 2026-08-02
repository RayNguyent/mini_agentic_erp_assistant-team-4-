"""Dynamic context assembly under an explicit token budget.

Everything that could go into a prompt — system policy, the user turn,
compacted conversation, working memory, long-term memory, retrieved document
chunks, tool results — is scored and offered to the budget in priority order.
What is dropped is recorded with a reason, not silently omitted: a budget that
runs out and drops the safety policy is indistinguishable from one that never
had it, unless the exclusion is logged (app.state.ContextBuildLog).

Priority order (highest first) is deliberate: policy and the live user turn
are never negotiable; retrieved evidence and tool results are the reason the
turn exists; memory is useful context that is allowed to lose first when the
budget is tight.
"""

from dataclasses import dataclass, field
from enum import IntEnum

from app.rag.documents import Chunk
from app.state import ContextBuildLog, ContextItem
from app.tokenizer import count_tokens

DEFAULT_BUDGET_TOKENS = 6000


class Priority(IntEnum):
    """Lower number = considered first. Ties keep insertion order."""

    SYSTEM_POLICY = 0
    USER_TURN = 1
    RETRIEVED_EVIDENCE = 2
    TOOL_RESULT = 3
    WORKING_MEMORY = 4
    LONG_TERM_MEMORY = 5
    CONVERSATION_HISTORY = 6


@dataclass(slots=True)
class ContextBlock:
    kind: str
    ref: str
    text: str
    priority: Priority
    tokens: int = 0

    def __post_init__(self) -> None:
        if not self.tokens:
            self.tokens = count_tokens(self.text)


@dataclass(slots=True)
class BuiltContext:
    system_prompt: str
    blocks: list[ContextBlock]  # included, in priority order
    log: ContextBuildLog

    def render(self) -> str:
        return "\n\n".join(b.text for b in self.blocks)


def system_policy_block(text: str) -> ContextBlock:
    return ContextBlock(kind="system_policy", ref="policy", text=text, priority=Priority.SYSTEM_POLICY)


def user_turn_block(message: str) -> ContextBlock:
    return ContextBlock(kind="user_turn", ref="current", text=f"User: {message}", priority=Priority.USER_TURN)


def evidence_blocks(chunks: list[Chunk]) -> list[ContextBlock]:
    return [
        ContextBlock(
            kind="retrieved_evidence",
            ref=chunk.chunk_id,
            text=f'<block id="{chunk.chunk_id}">{chunk.text}</block>',
            priority=Priority.RETRIEVED_EVIDENCE,
        )
        for chunk in chunks
    ]


def tool_result_block(tool_name: str, rendered: str) -> ContextBlock:
    return ContextBlock(
        kind="tool_result", ref=tool_name, text=f"Tool result ({tool_name}): {rendered}",
        priority=Priority.TOOL_RESULT,
    )


def working_memory_blocks(items: dict[str, object]) -> list[ContextBlock]:
    return [
        ContextBlock(kind="working_memory", ref=key, text=f"Session fact — {key}: {value}", priority=Priority.WORKING_MEMORY)
        for key, value in items.items()
    ]


def long_term_memory_blocks(entries) -> list[ContextBlock]:
    return [
        ContextBlock(kind="long_term_memory", ref=e.memory_id, text=f"Known fact: {e.text}", priority=Priority.LONG_TERM_MEMORY)
        for e in entries
    ]


def conversation_blocks(messages: list[dict]) -> list[ContextBlock]:
    return [
        ContextBlock(
            kind="conversation_history", ref=f"turn-{i}",
            text=f"{m.get('role', 'user')}: {m.get('content', '')}",
            priority=Priority.CONVERSATION_HISTORY,
        )
        for i, m in enumerate(messages)
    ]


def build_context(
    blocks: list[ContextBlock],
    *,
    budget_tokens: int = DEFAULT_BUDGET_TOKENS,
) -> BuiltContext:
    """Greedily admit blocks in priority order until the budget is spent.

    A lower-priority block that would fit is still skipped once a
    higher-priority block after it in the ordering has been rejected only if
    it doesn't fit either — i.e. this is priority-then-fit, not best-effort
    bin-packing. Predictability (policy always wins) matters more here than
    squeezing in one extra low-priority fact.
    """
    ordered = sorted(blocks, key=lambda b: b.priority)
    used = 0
    included: list[ContextBlock] = []
    log_items: list[ContextItem] = []

    for block in ordered:
        if used + block.tokens <= budget_tokens:
            included.append(block)
            used += block.tokens
            log_items.append(ContextItem(kind=block.kind, ref=block.ref, tokens=block.tokens, included=True))
        else:
            log_items.append(
                ContextItem(
                    kind=block.kind, ref=block.ref, tokens=block.tokens, included=False,
                    reason=f"budget exhausted ({used}/{budget_tokens} tokens used before this block)",
                )
            )

    # Restore something close to a natural reading order for the render:
    # policy and the user turn first, then evidence/tools, memory last.
    included.sort(key=lambda b: b.priority)

    system_blocks = [b for b in included if b.priority == Priority.SYSTEM_POLICY]
    system_prompt = "\n\n".join(b.text for b in system_blocks)
    body_blocks = [b for b in included if b.priority != Priority.SYSTEM_POLICY]

    return BuiltContext(
        system_prompt=system_prompt,
        blocks=body_blocks,
        log=ContextBuildLog(budget_tokens=budget_tokens, used_tokens=used, items=log_items),
    )
