"""Vector index port, with a Chroma adapter and a null adapter.

The retriever depends on this Protocol, never on `chromadb` — swapping Chroma
for Qdrant or pgvector is an adapter change, and the null adapter is what makes
"no embedding provider configured" an ordinary operating mode instead of an
outage.

Chroma is used in embedded persistent mode: no server, no Docker, a directory
on disk. We pass our own vectors rather than letting Chroma embed, so the
embedding provider stays behind our port (app/rag/embed.py) and the index never
silently reaches out to a network service of its own.

ACL note: the classification of each chunk is stored as Chroma metadata and the
allowed set is passed as a `where` filter, so restricted chunks are excluded by
the index itself rather than trimmed from the result afterwards. The retriever
still re-filters — defence in depth, since a bug in a `where` clause should not
be the only thing standing between a developer and the restricted corpus.
"""

import logging
import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable

from app.rag.documents import Chunk

logger = logging.getLogger(__name__)

CHROMA_DIR = Path("data/chroma")
COLLECTION = "project_chunks"


@runtime_checkable
class VectorIndex(Protocol):
    name: str
    available: bool

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        allowed_classifications: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return `(chunk_id, similarity)`, best first."""
        ...

    def count(self) -> int: ...

    def reset(self) -> None: ...


class NullVectorIndex:
    """No vector arm. Retrieval degrades to BM25-only, which still cites."""

    name = "none"
    available = False

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        return None

    def search(self, query_vector, top_k=20, allowed_classifications=None):
        return []

    def count(self) -> int:
        return 0

    def reset(self) -> None:
        return None


class ChromaVectorIndex:
    """Embedded Chroma with a local persist directory."""

    name = "chroma"
    available = True

    def __init__(self, persist_dir: Path | str = CHROMA_DIR, collection: str = COLLECTION) -> None:
        import chromadb
        from chromadb.config import Settings

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self._collection_name = collection
        self._client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection,
            # Cosine, to match the L2-normalised embeddings the adapters return.
            metadata={"hnsw:space": "cosine"},
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if not chunks or not vectors:
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"vector/chunk mismatch: {len(vectors)} vectors for {len(chunks)} chunks"
            )
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "title": c.title,
                    "heading": c.heading,
                    "classification": c.classification,
                    "project_code": c.project_code,
                }
                for c in chunks
            ],
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 20,
        allowed_classifications: list[str] | None = None,
    ) -> list[tuple[str, float]]:
        if not query_vector or self.count() == 0:
            return []

        where = (
            {"classification": {"$in": list(allowed_classifications)}}
            if allowed_classifications
            else None
        )
        try:
            response = self._collection.query(
                query_embeddings=[query_vector],
                n_results=min(top_k, self.count()),
                where=where,
                include=["distances"],
            )
        except Exception as exc:  # noqa: BLE001 - caller degrades to BM25
            logger.warning("chroma query failed: %s", exc)
            return []

        ids = (response.get("ids") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]
        # Chroma reports cosine *distance*; convert back to similarity and drop
        # anything at or below zero so an unrelated chunk never ranks.
        scored = [
            (chunk_id, 1.0 - float(distance))
            for chunk_id, distance in zip(ids, distances)
        ]
        return [pair for pair in scored if pair[1] > 0]

    def count(self) -> int:
        try:
            return self._collection.count()
        except Exception:  # noqa: BLE001
            return 0

    def reset(self) -> None:
        """Drop and recreate the collection so a rebuild cannot leave stale
        vectors paired with a fresh chunk list."""
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:  # noqa: BLE001 - absent collection is fine
            pass
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )

    def destroy(self) -> None:
        shutil.rmtree(self.persist_dir, ignore_errors=True)


def build_vector_index(
    persist_dir: Path | str = CHROMA_DIR, *, enabled: bool = True
) -> VectorIndex:
    """Chroma when enabled and importable, Null otherwise. Never raises —
    an unavailable vector index is a supported mode, not a startup failure."""
    if not enabled:
        return NullVectorIndex()
    try:
        return ChromaVectorIndex(persist_dir)
    except Exception as exc:  # noqa: BLE001
        logger.warning("chroma unavailable, using BM25-only retrieval: %s", exc)
        return NullVectorIndex()
