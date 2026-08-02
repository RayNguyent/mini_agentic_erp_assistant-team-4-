"""The agentic retrieve → grade → refine → answer → verify loop, and the
citation contract that gates it."""

import pytest

from app.rag.agentic import (
    MAX_ATTEMPTS,
    RagNode,
    RelevanceGrade,
    grade_deterministic,
    plan_query_deterministic,
    refine_query,
    run_agentic_rag,
)
from app.rag.answer import (
    GroundedAnswer,
    _extract_json,
    answer_from_documents,
    build_evidence_prompt,
    verify_citations,
)
from app.rag.documents import Chunk, ScoredChunk
from app.rag.retrieve import RetrievalDiagnostics, RetrievalResult, Retriever
from app.rag.store import ChunkStore
from app.rag.vector_index import NullVectorIndex


def make_chunk(chunk_id, text, classification="internal", **kw):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        title=kw.pop("title", "Doc"),
        text=text,
        classification=classification,
        **kw,
    )


def scored(chunk_id, text, **kw):
    return ScoredChunk(chunk=make_chunk(chunk_id, text, **kw), score=1.0)


def result_of(*scored_chunks) -> RetrievalResult:
    return RetrievalResult(list(scored_chunks), RetrievalDiagnostics(query="q"))


def retriever_over(chunks) -> Retriever:
    return Retriever(ChunkStore(chunks=chunks), vector_index=NullVectorIndex())


# --- query planning ----------------------------------------------------------


def test_planning_drops_interrogative_framing():
    assert "what" not in plan_query_deterministic("What is the approved budget?")


def test_planning_expands_user_wording_into_corpus_vocabulary():
    query = plan_query_deterministic("Is PRJ-001 going to be late?")
    assert "prj-001" in query
    assert "go-live" in query or "schedule" in query


def test_expansion_does_not_duplicate_terms_when_reapplied():
    once = plan_query_deterministic("Is PRJ-001 late?")
    twice = plan_query_deterministic(once)
    assert len(twice.split()) == len(set(twice.split()))


# --- grading -----------------------------------------------------------------


def test_empty_evidence_is_never_graded_sufficient():
    result = grade_deterministic("anything at all", [])
    assert result.sufficient is False
    assert result.answerable_on_partial is False


def test_evidence_covering_the_question_is_sufficient():
    result = grade_deterministic(
        "approved budget PRJ-001",
        [scored("B#c00", "the approved budget for PRJ-001 is one million USD")],
    )
    assert result.sufficient is True
    assert result.relevant_chunk_ids == ["B#c00"]


def test_off_topic_evidence_is_neither_sufficient_nor_answerable():
    result = grade_deterministic(
        "how many staff work in the Tokyo office",
        [scored("X#c00", "sprints are two weeks long and start on a Monday")],
    )
    assert result.sufficient is False
    assert result.answerable_on_partial is False


def test_a_chunk_sharing_one_incidental_word_is_not_counted_as_relevant():
    # The failure this guards: an aggregate score dragged over the line by a
    # chunk that merely mentions one of the question's words.
    result = grade_deterministic(
        "licensing negotiation target reduction",
        [scored("S#c00", "support response targets are four hours for severity two")],
    )
    assert result.relevant_chunk_ids == []
    assert result.answerable_on_partial is False


def test_grade_reports_the_terms_it_could_not_cover():
    result = grade_deterministic(
        "budget helicopter", [scored("B#c00", "the budget report covers actual cost")]
    )
    assert "helicopter" in result.missing


# --- refinement --------------------------------------------------------------


def _rag_state(**kw):
    from app.rag.agentic import RagState

    return RagState(question=kw.pop("question", "what is the sprint velocity"), **kw)


def test_refinement_folds_in_the_missing_terms():
    state = _rag_state(
        queries_tried=["sprint velocity"],
        grade=RelevanceGrade(missing="cadence"),
    )
    assert "cadence" in refine_query(state)


def test_refinement_prefers_an_explicit_suggestion_from_the_grader():
    state = _rag_state(
        queries_tried=["sprint velocity"],
        grade=RelevanceGrade(suggested_query="delivery iteration milestone"),
    )
    assert refine_query(state) == "delivery iteration milestone"


def test_refinement_never_repeats_a_query_already_tried():
    state = _rag_state(
        queries_tried=["sprint velocity", "delivery iteration"],
        grade=RelevanceGrade(suggested_query="delivery iteration"),
    )
    assert refine_query(state) not in ["delivery iteration"]


# --- the loop ----------------------------------------------------------------
#
# These pass an explicit role. With role=None the ACL grants public-only
# clearance (least privilege for an unidentified caller), so an internal chunk
# would be filtered before the loop ever graded it — the run would refuse for
# an access reason rather than an evidence one.


def test_a_well_covered_question_answers_on_the_first_attempt():
    retriever = retriever_over(
        [make_chunk("B#c00", "the approved budget for PRJ-001 is 1,200,000 USD")]
    )
    state = run_agentic_rag(
        "What is the approved budget for PRJ-001?", retriever, role="developer"
    )

    assert state.attempt == 1
    assert state.refused is False
    assert [c.chunk_id for c in state.citations] == ["B#c00"]
    assert state.next_action == RagNode.DONE


def test_an_unidentified_caller_gets_public_clearance_only():
    retriever = retriever_over([make_chunk("I#c00", "internal delivery cadence detail")])
    assert run_agentic_rag("delivery cadence detail", retriever).refused is True
    assert (
        run_agentic_rag("delivery cadence detail", retriever, role="developer").refused
        is False
    )


def test_a_thin_first_result_triggers_a_bounded_refinement_loop():
    retriever = retriever_over([make_chunk("S#c00", "sprints are two weeks long")])
    state = run_agentic_rag(
        "How many staff work in the Tokyo office?", retriever, role="developer"
    )

    assert 1 < state.attempt <= MAX_ATTEMPTS
    assert len(state.queries_tried) == state.attempt
    assert state.refused is True


def test_the_loop_terminates_when_refinement_has_nothing_new_to_try():
    # Regression: refine -> verify -> refine is a cycle neither node counts as
    # an attempt, so only an explicit exhausted flag ends it.
    retriever = retriever_over([make_chunk("S#c00", "unrelated content entirely")])
    state = run_agentic_rag("zzz", retriever, role="developer")
    assert state.next_action == RagNode.DONE
    assert state.refused is True


def test_an_empty_corpus_refuses_rather_than_erroring():
    state = run_agentic_rag("What is the budget?", retriever_over([]), role="auditor")
    assert state.refused is True
    assert state.citations == []


def test_every_attempt_records_its_own_diagnostics_entry():
    retriever = retriever_over([make_chunk("S#c00", "sprints are two weeks long")])
    state = run_agentic_rag(
        "How many staff work in the Tokyo office?", retriever, role="developer"
    )
    assert len(state.diagnostics) == state.attempt


def test_acl_is_enforced_through_the_whole_loop_not_just_the_first_pass():
    retriever = retriever_over(
        [make_chunk("SEC#c00", "licensing negotiation target reduction", classification="restricted")]
    )
    denied = run_agentic_rag("licensing negotiation target", retriever, role="developer")
    allowed = run_agentic_rag("licensing negotiation target", retriever, role="auditor")

    assert denied.refused is True and denied.citations == []
    assert allowed.refused is False
    assert [c.chunk_id for c in allowed.citations] == ["SEC#c00"]


def test_refusal_answer_is_the_standard_refusal_text_not_a_guess():
    from app.rag.answer import REFUSAL_TEXT

    state = run_agentic_rag("what is the weather in Paris", retriever_over([]), role="auditor")
    assert state.answer == REFUSAL_TEXT


# --- grounded answer contract ------------------------------------------------


def test_a_sufficient_answer_without_citations_cannot_be_constructed():
    with pytest.raises(ValueError, match="must cite at least one"):
        GroundedAnswer(answer="The budget is 1.2M", citations=[], sufficient=True)


def test_a_refusal_needs_no_citations():
    assert GroundedAnswer(sufficient=False).refused is True


def test_sufficient_true_requires_a_non_empty_answer():
    with pytest.raises(ValueError, match="requires a non-empty answer"):
        GroundedAnswer(answer="   ", citations=["A#c00"], sufficient=True)


def test_fabricated_citation_ids_are_dropped():
    retrieval = result_of(scored("REAL#c00", "real evidence text"))
    answer = GroundedAnswer(
        answer="grounded claim", citations=["REAL#c00", "MADE-UP#c99"], sufficient=True
    )
    verified, citations = verify_citations(answer, retrieval)

    assert verified.citations == ["REAL#c00"]
    assert [c.chunk_id for c in citations] == ["REAL#c00"]


def test_an_answer_citing_only_fabricated_ids_is_demoted_to_a_refusal():
    retrieval = result_of(scored("REAL#c00", "real evidence"))
    answer = GroundedAnswer(answer="claim", citations=["FAKE#c01"], sufficient=True)
    verified, citations = verify_citations(answer, retrieval)

    assert verified.refused is True
    assert citations == []


def test_evidence_is_wrapped_in_untrusted_delimiters_separate_from_the_question():
    prompt, log, included = build_evidence_prompt("What is the budget?", result_of(scored("B#c00", "1.2M")))
    assert 'untrusted="true"' in prompt
    assert 'id="B#c00"' in prompt
    assert prompt.index("<evidence") < prompt.index("Question:")
    assert [s.chunk.chunk_id for s in included] == ["B#c00"]
    assert log.excluded == []


def test_a_chunk_that_exceeds_the_budget_alone_is_excluded_and_logged():
    huge = scored("B#c00", "word " * 5000)  # far larger than any small budget
    prompt, log, included = build_evidence_prompt("q", result_of(huge), budget_tokens=10)
    assert included == []
    assert log.excluded and log.excluded[0].ref == "B#c00"
    assert "budget exhausted" in log.excluded[0].reason


def test_empty_retrieval_refuses_without_calling_a_provider():
    class _Exploding:
        def generate(self, *a, **kw):
            raise AssertionError("provider must not be called with no evidence")

    answer, citations, log = answer_from_documents("q", result_of(), _Exploding())
    assert answer.refused is True and citations == []
    assert log.used_tokens == 0


def test_offline_mode_produces_a_cited_extractive_answer():
    answer, citations, log = answer_from_documents(
        "What is the budget?", result_of(scored("B#c00", "The approved budget is 1.2M USD"))
    )
    assert answer.refused is False
    assert [c.chunk_id for c in citations] == ["B#c00"]
    assert "B#c00" in answer.answer
    assert log.used_tokens > 0


def test_a_chunk_excluded_by_the_budget_is_never_offered_to_the_provider():
    class _RecordingProvider:
        def __init__(self):
            self.seen_prompt = None

        def generate(self, prompt, *, system=None, history=None):
            self.seen_prompt = prompt
            return '{"answer": "the budget is X", "citations": ["A#c00"], "sufficient": true}'

    provider = _RecordingProvider()
    retrieval = result_of(scored("A#c00", "small evidence"), scored("B#c01", "word " * 2000))
    answer, citations, log = answer_from_documents("q", retrieval, provider, budget_tokens=100)

    assert "B#c01" not in provider.seen_prompt
    assert any(item.ref == "B#c01" and not item.included for item in log.items)


def test_a_provider_citing_a_budget_excluded_chunk_is_treated_as_fabricated():
    # The chunk exists in the full retrieval set, so a naive "was it
    # retrieved at all" check would wrongly accept this citation — it must be
    # checked against what was actually *shown*, not what was retrieved.
    class _CitesExcluded:
        def generate(self, prompt, *, system=None, history=None):
            return '{"answer": "claim", "citations": ["B#c01"], "sufficient": true}'

    retrieval = result_of(scored("A#c00", "small evidence"), scored("B#c01", "word " * 2000))
    answer, citations, _ = answer_from_documents("q", retrieval, _CitesExcluded(), budget_tokens=100)

    assert answer.refused is True
    assert citations == []


class _FakeProvider:
    def __init__(self, response):
        self._response = response

    def generate(self, prompt, *, system=None, history=None):
        return self._response


def test_a_provider_refusal_is_honoured():
    answer, citations, _ = answer_from_documents(
        "q",
        result_of(scored("B#c00", "some evidence")),
        _FakeProvider('{"answer": "", "citations": [], "sufficient": false}'),
    )
    assert answer.refused is True and citations == []


def test_malformed_provider_output_falls_back_to_the_extractive_path():
    answer, _, _ = answer_from_documents(
        "q", result_of(scored("B#c00", "evidence")), _FakeProvider("not json at all")
    )
    assert answer.refused is False
    assert answer.citations == ["B#c00"]


def test_a_provider_answer_citing_nothing_it_was_shown_becomes_a_refusal():
    answer, _, _ = answer_from_documents(
        "q",
        result_of(scored("B#c00", "evidence")),
        _FakeProvider('{"answer": "confident claim", "citations": ["GHOST#c00"], "sufficient": true}'),
    )
    assert answer.refused is True


@pytest.mark.parametrize(
    "raw",
    [
        '{"answer": "a", "citations": ["X"], "sufficient": true}',
        '```json\n{"answer": "a", "citations": ["X"], "sufficient": true}\n```',
        'Sure! {"answer": "a", "citations": ["X"], "sufficient": true} Hope that helps.',
    ],
)
def test_json_is_recovered_from_fenced_or_chatty_output(raw):
    assert _extract_json(raw)["answer"] == "a"
