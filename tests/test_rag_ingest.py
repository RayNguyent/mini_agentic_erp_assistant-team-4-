"""Ingestion, the chunk store, and the Chroma vector-index adapter."""

import json

import pytest

from app.rag.documents import Chunk
from app.rag.ingest import ingest
from app.rag.store import ChunkStore
from app.rag.vector_index import ChromaVectorIndex, NullVectorIndex, build_vector_index

CORPUS = """---
doc_id: DOC-T1
title: Test Charter
project_code: PRJ-009
classification: internal
updated_at: 2026-01-01
---

# Charter

## Purpose

This project replaces the legacy stack with a hosted platform.

## Scope

In scope is the general ledger migration and vendor onboarding work.
"""

FINANCE_CORPUS = """---
doc_id: DOC-T2
title: Test Budget
project_code: PRJ-009
classification: finance
updated_at: 2026-01-02
---

# Budget

## Position

The approved budget is 500,000 USD with 200,000 USD spent to date.
"""


@pytest.fixture
def corpus_dir(tmp_path):
    directory = tmp_path / "corpus"
    directory.mkdir()
    (directory / "charter.md").write_text(CORPUS, encoding="utf-8")
    (directory / "budget.md").write_text(FINANCE_CORPUS, encoding="utf-8")
    return directory


# --- ingestion ---------------------------------------------------------------


def test_ingest_builds_chunks_and_reports_per_document_detail(corpus_dir, tmp_path):
    store, report = ingest(
        corpus_dir, tmp_path / "index", vector_index=NullVectorIndex(), log_path=None
    )

    assert report["documents"] == 2
    assert store.size >= 2
    assert {d["doc_id"] for d in report["documents_detail"]} == {"DOC-T1", "DOC-T2"}
    assert all(d["tokens"] > 0 for d in report["documents_detail"])


def test_ingest_preserves_classification_onto_every_chunk(corpus_dir, tmp_path):
    store, _ = ingest(
        corpus_dir, tmp_path / "index", vector_index=NullVectorIndex(), log_path=None
    )
    finance = [c for c in store.chunks if c.doc_id == "DOC-T2"]
    assert finance and all(c.classification == "finance" for c in finance)


def test_ingest_writes_an_append_only_log(corpus_dir, tmp_path):
    log = tmp_path / "ingestion_log.jsonl"
    for _ in range(2):
        ingest(corpus_dir, tmp_path / "index", vector_index=NullVectorIndex(), log_path=log)

    entries = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert entries[0]["chunks"] == entries[1]["chunks"]


def test_ingest_without_embeddings_reports_a_keyword_only_index(corpus_dir, tmp_path):
    _, report = ingest(
        corpus_dir, tmp_path / "index", vector_index=NullVectorIndex(), log_path=None
    )
    assert report["vectors"] == 0
    assert report["embedding_error"] is None


def test_missing_corpus_directory_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="corpus directory not found"):
        ingest(tmp_path / "nope", tmp_path / "index", log_path=None)


# --- store round-trip ---------------------------------------------------------


def test_store_round_trips_through_disk(corpus_dir, tmp_path):
    index_dir = tmp_path / "index"
    original, _ = ingest(corpus_dir, index_dir, vector_index=NullVectorIndex(), log_path=None)

    reloaded = ChunkStore.load(index_dir)
    assert [c.chunk_id for c in reloaded.chunks] == [c.chunk_id for c in original.chunks]
    assert reloaded.manifest["documents"] == 2


def test_loading_a_missing_index_yields_an_empty_store_not_an_error(tmp_path):
    store = ChunkStore.load(tmp_path / "absent")
    assert store.size == 0
    assert store.doc_ids == []


def test_lookup_by_chunk_id():
    chunk = Chunk(chunk_id="A#c00", doc_id="A", title="T", text="body")
    store = ChunkStore(chunks=[chunk])
    assert store.by_id("A#c00") is chunk
    assert store.by_id("missing") is None


# --- Chroma adapter -----------------------------------------------------------


@pytest.fixture
def chroma(tmp_path):
    index = ChromaVectorIndex(tmp_path / "chroma", collection="test_chunks")
    yield index
    index.destroy()


def _chunk(chunk_id, classification="internal"):
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        title="Doc",
        text=f"body of {chunk_id}",
        classification=classification,
    )


def test_chroma_upsert_then_search_ranks_the_nearest_chunk_first(chroma):
    chroma.upsert([_chunk("A#c00"), _chunk("B#c00")], [[1.0, 0.0], [0.6, 0.8]])

    hits = chroma.search([1.0, 0.0], top_k=2)
    assert [cid for cid, _ in hits] == ["A#c00", "B#c00"]
    assert hits[0][1] > hits[1][1]  # similarity, so higher is nearer


def test_an_orthogonal_vector_is_dropped_rather_than_returned_as_a_weak_match(chroma):
    # Cosine similarity 0 means "shares nothing with the query". Returning it
    # would pad the evidence set with a chunk the grader then has to reject.
    chroma.upsert([_chunk("A#c00"), _chunk("ORTH#c00")], [[1.0, 0.0], [0.0, 1.0]])
    assert [cid for cid, _ in chroma.search([1.0, 0.0], top_k=5)] == ["A#c00"]


def test_chroma_filters_by_classification_in_the_query_itself(chroma):
    chroma.upsert(
        [_chunk("PUB#c00", "public"), _chunk("SEC#c00", "restricted")],
        [[1.0, 0.0], [1.0, 0.0]],
    )
    ids = [cid for cid, _ in chroma.search([1.0, 0.0], allowed_classifications=["public"])]
    assert ids == ["PUB#c00"]


def test_chroma_reset_clears_the_collection(chroma):
    chroma.upsert([_chunk("A#c00")], [[1.0, 0.0]])
    assert chroma.count() == 1
    chroma.reset()
    assert chroma.count() == 0
    assert chroma.search([1.0, 0.0]) == []


def test_chroma_rejects_a_vector_count_that_does_not_match_the_chunks(chroma):
    with pytest.raises(ValueError, match="vector/chunk mismatch"):
        chroma.upsert([_chunk("A#c00"), _chunk("B#c00")], [[1.0, 0.0]])


def test_upsert_replaces_rather_than_duplicates_an_existing_id(chroma):
    chroma.upsert([_chunk("A#c00")], [[1.0, 0.0]])
    chroma.upsert([_chunk("A#c00")], [[0.0, 1.0]])
    assert chroma.count() == 1


def test_build_vector_index_returns_null_when_disabled():
    assert isinstance(build_vector_index(enabled=False), NullVectorIndex)


def test_null_index_is_inert_and_reports_unavailable():
    index = NullVectorIndex()
    index.upsert([_chunk("A#c00")], [[1.0]])
    assert index.available is False
    assert index.count() == 0
    assert index.search([1.0]) == []
