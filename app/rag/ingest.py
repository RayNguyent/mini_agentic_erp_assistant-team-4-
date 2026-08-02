"""Corpus ingestion: load -> normalize -> chunk -> embed -> persist.

Run as a module:

    uv run python -m app.rag.ingest            # build the index
    uv run python -m app.rag.ingest --rebuild  # discard and rebuild
    uv run python -m app.rag.ingest --stats    # report without writing

Writes an append-only ingestion log to data/ingestion_log.jsonl. The log is
submission evidence: it records what was ingested, how it chunked, and whether
embeddings were available for that run.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.rag.chunk import chunk_corpus
from app.rag.documents import CORPUS_DIR, load_corpus
from app.rag.embed import EmbeddingProvider, build_embedding_provider
from app.rag.store import INDEX_DIR, ChunkStore
from app.rag.vector_index import VectorIndex, build_vector_index

INGESTION_LOG = Path("data/ingestion_log.jsonl")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def ingest(
    corpus_dir: Path | str = CORPUS_DIR,
    index_dir: Path | str = INDEX_DIR,
    embedder: EmbeddingProvider | None = None,
    vector_index: VectorIndex | None = None,
    *,
    write: bool = True,
    log_path: Path | str | None = INGESTION_LOG,
) -> tuple[ChunkStore, dict]:
    """Build the chunk store and (when embeddings are available) the vector
    index. Returns `(store, report)`."""
    started = _now()
    documents = load_corpus(corpus_dir)
    chunks = chunk_corpus(documents)

    embedder = embedder or build_embedding_provider()
    if vector_index is None:
        vector_index = build_vector_index(enabled=embedder.available)

    vector_count = 0
    embedding_error = None
    if write and embedder.available and vector_index.available and chunks:
        try:
            vectors = embedder.embed_documents([c.text for c in chunks])
            # Replace wholesale: a partial upsert would leave chunks from a
            # previous corpus revision searchable alongside the new ones.
            vector_index.reset()
            vector_index.upsert(chunks, vectors)
            vector_count = vector_index.count()
        except Exception as exc:  # noqa: BLE001 - a keyword-only index is still valid
            embedding_error = f"{type(exc).__name__}: {exc}"

    manifest = {
        "built_at": started,
        "corpus_dir": str(corpus_dir).replace("\\", "/"),
        "documents": len(documents),
        "chunks": len(chunks),
        "embedding_provider": embedder.name,
        "vector_index": vector_index.name,
        "vectors": vector_count,
        "embedding_error": embedding_error,
    }
    store = ChunkStore(chunks=chunks, manifest=manifest)

    report = {
        **manifest,
        "finished_at": _now(),
        "total_tokens": sum(c.tokens for c in chunks),
        "documents_detail": [
            {
                "doc_id": d.doc_id,
                "title": d.title,
                "project_code": d.project_code,
                "classification": d.classification,
                "updated_at": d.updated_at,
                "chars": len(d.text),
                "chunks": sum(1 for c in chunks if c.doc_id == d.doc_id),
                "tokens": sum(c.tokens for c in chunks if c.doc_id == d.doc_id),
            }
            for d in documents
        ],
    }

    if write:
        store.save(index_dir)
        if log_path is not None:
            path = Path(log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(report) + "\n")

    return store, report


def _print_report(report: dict) -> None:
    print(f"corpus:     {report['corpus_dir']}")
    print(f"documents:  {report['documents']}")
    print(f"chunks:     {report['chunks']} ({report['total_tokens']} tokens)")
    print(
        f"embeddings: {report['embedding_provider']} -> "
        f"{report['vector_index']} ({report['vectors']} vectors)"
    )
    if report.get("embedding_error"):
        print(f"  ! embedding failed, keyword-only index: {report['embedding_error']}")
    print()
    print(f"{'doc_id':<20} {'class':<11} {'proj':<8} {'chunks':>6} {'tokens':>7}  title")
    for doc in report["documents_detail"]:
        print(
            f"{doc['doc_id']:<20} {doc['classification']:<11} {doc['project_code']:<8} "
            f"{doc['chunks']:>6} {doc['tokens']:>7}  {doc['title'][:48]}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.rag.ingest", description=__doc__)
    parser.add_argument("--corpus", default=str(CORPUS_DIR))
    parser.add_argument("--index", default=str(INDEX_DIR))
    parser.add_argument(
        "--rebuild", action="store_true", help="delete the existing index first"
    )
    parser.add_argument(
        "--stats", action="store_true", help="report only, do not write the index"
    )
    args = parser.parse_args(argv)

    if args.rebuild:
        index_dir = Path(args.index)
        if index_dir.exists():
            for path in index_dir.iterdir():
                path.unlink()
            print(f"cleared {index_dir}")
        # The vector collection is dropped inside ingest() before upsert; this
        # only clears it for the case where embeddings are now unavailable and
        # ingest() would otherwise leave the previous run's vectors in place.
        build_vector_index(enabled=True).reset()

    try:
        _, report = ingest(args.corpus, args.index, write=not args.stats)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_report(report)
    if not args.stats:
        print(f"\nindex written to {args.index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
