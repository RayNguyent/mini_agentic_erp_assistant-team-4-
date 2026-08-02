"""Token-budgeted context assembly: priority order, and the included/excluded
log that makes a dropped block visible instead of silent."""

from app.context.builder import (
    Priority,
    build_context,
    evidence_blocks,
    system_policy_block,
    user_turn_block,
    working_memory_blocks,
)
from app.rag.documents import Chunk


def _text_block(kind, ref, tokens, priority):
    from app.context.builder import ContextBlock

    return ContextBlock(kind=kind, ref=ref, text="w " * tokens, priority=priority, tokens=tokens)


def test_everything_fits_when_the_budget_is_generous():
    blocks = [system_policy_block("policy"), user_turn_block("hello")]
    built = build_context(blocks, budget_tokens=1000)
    assert built.log.excluded == []
    assert built.system_prompt == "policy"


def test_policy_is_never_the_block_that_gets_dropped():
    # Evidence is offered first in the input list (a plausible real ordering)
    # but policy must still win the budget purely on priority.
    policy = _text_block("system_policy", "policy", 50, Priority.SYSTEM_POLICY)
    huge_evidence = _text_block("retrieved_evidence", "E1", 500, Priority.RETRIEVED_EVIDENCE)
    built = build_context([huge_evidence, policy], budget_tokens=60)

    assert built.system_prompt == "w " * 50
    assert not any(i.included for i in built.log.items if i.kind == "retrieved_evidence")


def test_a_dropped_block_is_logged_with_a_reason_not_silently_omitted():
    small = _text_block("working_memory", "fact-1", 10, Priority.WORKING_MEMORY)
    over_budget = _text_block("working_memory", "fact-2", 1000, Priority.WORKING_MEMORY)
    built = build_context([small, over_budget], budget_tokens=20)

    excluded = built.log.excluded
    assert len(excluded) == 1
    assert excluded[0].ref == "fact-2"
    assert "budget exhausted" in excluded[0].reason


def test_higher_priority_blocks_are_admitted_before_lower_priority_ones():
    memory_fact = _text_block("working_memory", "mem", 40, Priority.WORKING_MEMORY)
    evidence = _text_block("retrieved_evidence", "ev", 40, Priority.RETRIEVED_EVIDENCE)
    # Budget fits only one of the two — evidence must win.
    built = build_context([memory_fact, evidence], budget_tokens=45)

    included_refs = {b.ref for b in built.blocks}
    assert "ev" in included_refs
    assert "mem" not in included_refs


def test_used_tokens_matches_the_sum_of_included_block_tokens():
    a = _text_block("tool_result", "a", 30, Priority.TOOL_RESULT)
    b = _text_block("tool_result", "b", 30, Priority.TOOL_RESULT)
    built = build_context([a, b], budget_tokens=50)
    assert built.log.used_tokens == sum(item.tokens for item in built.log.items if item.included)
    assert built.log.used_tokens <= 50


def test_evidence_blocks_carry_the_chunk_id_as_ref_for_traceability():
    chunk = Chunk(chunk_id="D#c00", doc_id="D", title="T", text="body")
    blocks = evidence_blocks([chunk])
    assert blocks[0].ref == "D#c00"
    assert blocks[0].priority == Priority.RETRIEVED_EVIDENCE


def test_working_memory_blocks_render_one_per_key():
    blocks = working_memory_blocks({"active_project_code": "PRJ-001", "last_tool": "list_risks"})
    assert {b.ref for b in blocks} == {"active_project_code", "last_tool"}


def test_render_joins_included_blocks_in_priority_order():
    built = build_context(
        [user_turn_block("what is the status?"), system_policy_block("be helpful")],
        budget_tokens=1000,
    )
    # system policy is reported separately via system_prompt, not in render()
    assert "what is the status?" in built.render()
    assert "be helpful" not in built.render()
