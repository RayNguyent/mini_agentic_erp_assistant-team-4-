"""Chunking, keyword scoring, ACL filtering, and hybrid fusion."""

import pytest

from app.rag.acl import can_read, clearance_for, filter_chunks
from app.rag.bm25 import BM25Index, tokenize
from app.rag.chunk import chunk_document
from app.rag.documents import Chunk, Document, normalize_text, parse_front_matter
from app.rag.retrieve import Retriever
from app.rag.store import ChunkStore
from app.rag.vector_index import NullVectorIndex


def make_chunk(chunk_id="D#c00", text="text", classification="internal", doc_id="D", **kw):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        title=kw.pop("title", "Doc"),
        text=text,
        classification=classification,
        **kw,
    )


# --- front matter + normalisation -------------------------------------------


def test_front_matter_is_split_from_the_body():
    meta, body = parse_front_matter(
        "---\ndoc_id: DOC-1\ntitle: A Title\n---\n# Heading\n\nBody text.\n"
    )
    assert meta == {"doc_id": "DOC-1", "title": "A Title"}
    assert body.startswith("# Heading")


def test_document_without_front_matter_is_returned_unchanged():
    meta, body = parse_front_matter("# Just a heading\n\nBody.")
    assert meta == {}
    assert body == "# Just a heading\n\nBody."


def test_unknown_classification_is_rejected_at_load():
    with pytest.raises(ValueError, match="unknown classification"):
        Document(doc_id="D", title="T", classification="top-secret", text="x")


def test_normalisation_strips_blockquote_markers_and_collapses_blank_runs():
    assert normalize_text("> quoted line\n\n\n\nnext") == "quoted line\n\nnext"


# --- chunking ----------------------------------------------------------------


def _doc(text: str) -> Document:
    return Document(doc_id="DOC-T", title="Test Doc", classification="internal", text=text)


def test_chunks_carry_heading_provenance():
    chunks = chunk_document(_doc("# Top\n\n## Alpha\n\nBody about alpha.\n"))
    assert chunks
    assert "Alpha" in chunks[0].heading


def test_short_sections_are_merged_rather_than_emitted_as_fragments():
    text = "\n\n".join(f"## Section {i}\n\nA short paragraph number {i}." for i in range(8))
    chunks = chunk_document(_doc(text))
    # Eight tiny sections must not become eight chunks.
    assert len(chunks) < 8
    assert all(c.tokens > 0 for c in chunks)


def test_chunk_ids_are_unique_and_namespaced_by_document():
    chunks = chunk_document(_doc("## A\n\n" + "word " * 400 + "\n\n## B\n\n" + "other " * 400))
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(cid.startswith("DOC-T#c") for cid in ids)


def test_chunks_inherit_document_classification():
    doc = Document(doc_id="D", title="T", classification="restricted", text="## H\n\nBody here.")
    assert all(c.classification == "restricted" for c in chunk_document(doc))


# --- BM25 ---------------------------------------------------------------------


def test_identifiers_survive_tokenisation_as_single_terms():
    assert "prj-001" in tokenize("Status of PRJ-001 please")


def test_stopwords_are_dropped_but_domain_terms_are_kept():
    terms = tokenize("What is the risk and the budget for the open sprint")
    assert "the" not in terms and "is" not in terms
    assert {"risk", "budget", "open", "sprint"} <= set(terms)


def test_bm25_ranks_the_document_that_actually_discusses_the_term():
    index = BM25Index(
        [
            "the budget report covers actual cost and variance",
            "the sprint plan covers velocity and cadence",
            "unrelated vendor availability text",
        ]
    )
    assert index.search("budget variance", top_k=1)[0][0] == 0


def test_bm25_returns_nothing_for_a_term_absent_from_the_corpus():
    index = BM25Index(["budget and cost", "sprint and velocity"])
    assert index.search("tokyo helicopter") == []


def test_bm25_handles_an_empty_corpus():
    assert BM25Index([]).search("anything") == []


# --- ACL ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "role,classification,expected",
    [
        ("developer", "public", True),
        ("developer", "internal", True),
        ("developer", "finance", False),
        ("developer", "restricted", False),
        ("project_manager", "finance", True),
        ("project_manager", "restricted", False),
        ("auditor", "restricted", True),
    ],
)
def test_clearance_matrix(role, classification, expected):
    assert can_read(role, classification) is expected


def test_unknown_or_missing_role_gets_least_privilege_not_most():
    assert clearance_for(None) == frozenset({"public"})
    assert clearance_for("wizard") == frozenset({"public"})


def test_filter_reports_what_it_denied_so_the_trace_can_prove_it_ran():
    chunks = [
        make_chunk("A", classification="public"),
        make_chunk("B", classification="finance"),
    ]
    visible, denied = filter_chunks(chunks, "developer")
    assert [c.chunk_id for c in visible] == ["A"]
    assert denied == ["B"]


# --- retriever ----------------------------------------------------------------


def _retriever(chunks: list[Chunk]) -> Retriever:
    return Retriever(ChunkStore(chunks=chunks), vector_index=NullVectorIndex())


def test_retrieval_excludes_chunks_the_role_may_not_read():
    retriever = _retriever(
        [
            make_chunk("PUB#c00", text="the approved budget is one million", classification="public"),
            make_chunk("FIN#c00", text="the approved budget is one million", classification="finance"),
        ]
    )
    result = retriever.retrieve("approved budget", role="developer")
    assert [s.chunk.chunk_id for s in result.results] == ["PUB#c00"]
    assert "FIN#c00" in result.diagnostics.acl_denied


def test_a_higher_clearance_sees_the_restricted_chunk_for_the_same_query():
    retriever = _retriever(
        [make_chunk("SEC#c00", text="licensing negotiation target", classification="restricted")]
    )
    assert _ids(retriever.retrieve("licensing negotiation", role="auditor")) == ["SEC#c00"]
    assert _ids(retriever.retrieve("licensing negotiation", role="developer")) == []


def _ids(result):
    return [s.chunk.chunk_id for s in result.results]


def test_empty_index_returns_no_results_with_an_explanatory_note():
    result = _retriever([]).retrieve("anything", role="auditor")
    assert result.is_empty
    assert any("index is empty" in note for note in result.diagnostics.notes)


def test_bm25_only_mode_is_reported_as_degraded_but_still_returns_citations():
    retriever = _retriever([make_chunk("A#c00", text="sprint velocity and cadence")])
    result = retriever.retrieve("sprint velocity", role="developer")
    assert result.results
    assert result.diagnostics.degraded is True
    assert result.citations()[0].chunk_id == "A#c00"


def test_project_filter_narrows_but_keeps_corpus_wide_documents():
    retriever = _retriever(
        [
            make_chunk("P1#c00", text="delivery risk register", project_code="PRJ-001"),
            make_chunk("P2#c00", text="delivery risk register", project_code="PRJ-002"),
            make_chunk("AL#c00", text="delivery risk register", project_code="ALL"),
        ]
    )
    ids = _ids(retriever.retrieve("risk register", role="developer", project_code="PRJ-001"))
    assert set(ids) == {"P1#c00", "AL#c00"}


def test_a_project_filter_cannot_widen_what_a_role_may_see():
    retriever = _retriever(
        [make_chunk("F#c00", text="budget variance", classification="finance", project_code="PRJ-001")]
    )
    result = retriever.retrieve("budget variance", role="developer", project_code="PRJ-001")
    assert result.is_empty


def test_diagnostics_record_the_stage_counts():
    retriever = _retriever([make_chunk("A#c00", text="sprint velocity cadence")])
    diagnostics = retriever.retrieve("sprint velocity", role="developer").diagnostics
    assert diagnostics.bm25_hits == 1
    assert diagnostics.vector_hits == 0
    assert diagnostics.returned == 1
    assert diagnostics.as_dict()["acl_denied_count"] == 0


class _StubVectorIndex:
    """Returns a fixed ranking so fusion can be tested without embeddings."""

    name = "stub"
    available = True

    def __init__(self, hits):
        self._hits = hits

    def upsert(self, chunks, vectors):
        return None

    def search(self, query_vector, top_k=20, allowed_classifications=None):
        return self._hits

    def count(self):
        return len(self._hits)

    def reset(self):
        return None


class _StubEmbedder:
    name = "stub"
    dimensions = 3
    available = True

    def embed_documents(self, texts):
        return [[1.0, 0.0, 0.0] for _ in texts]

    def embed_query(self, text):
        return [1.0, 0.0, 0.0]


def test_a_chunk_found_by_both_arms_outranks_one_found_by_a_single_arm():
    chunks = [
        make_chunk("KEY#c00", text="sprint velocity cadence"),
        make_chunk("VEC#c00", text="entirely unrelated wording"),
    ]
    retriever = Retriever(
        ChunkStore(chunks=chunks),
        embedder=_StubEmbedder(),
        vector_index=_StubVectorIndex([("VEC#c00", 0.9), ("KEY#c00", 0.8)]),
    )
    results = retriever.retrieve("sprint velocity", role="developer")

    top = results.results[0]
    assert top.chunk.chunk_id == "KEY#c00"
    assert set(top.matched_by) == {"bm25", "vector"}
    assert top.bm25_rank == 1 and top.vector_rank == 2


def test_a_failing_vector_arm_degrades_to_keyword_only_instead_of_erroring():
    class _Broken(_StubVectorIndex):
        def search(self, *a, **kw):
            raise RuntimeError("index unreachable")

    retriever = Retriever(
        ChunkStore(chunks=[make_chunk("A#c00", text="sprint velocity")]),
        embedder=_StubEmbedder(),
        vector_index=_Broken([]),
    )
    result = retriever.retrieve("sprint velocity", role="developer")
    assert result.results
    assert result.diagnostics.degraded is True
