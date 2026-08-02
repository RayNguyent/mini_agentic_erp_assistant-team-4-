"""The canonical chunk record store.

Deliberately separate from the vector index (app/rag/vector_index.py). Chunk
text and metadata must exist unconditionally — BM25 reads them, citations
render from them, and the deterministic offline profile has no embeddings at
all — whereas vectors exist only when an embedding provider is configured.
Collapsing the two would make the keyword arm depend on the vector arm being
available, which is exactly the coupling the offline guarantee forbids.
"""

import json
from pathlib import Path

from app.rag.documents import Chunk

INDEX_DIR = Path("data/index")
_CHUNKS_FILE = "chunks.json"
_MANIFEST_FILE = "manifest.json"


class ChunkStore:
    """Chunk records plus the manifest describing how they were built."""

    def __init__(self, chunks: list[Chunk] | None = None, manifest: dict | None = None) -> None:
        self.chunks = chunks or []
        self.manifest = manifest or {}
        self._by_id = {chunk.chunk_id: chunk for chunk in self.chunks}

    @property
    def size(self) -> int:
        return len(self.chunks)

    @property
    def doc_ids(self) -> list[str]:
        return sorted({chunk.doc_id for chunk in self.chunks})

    def by_id(self, chunk_id: str) -> Chunk | None:
        return self._by_id.get(chunk_id)

    def texts(self) -> list[str]:
        return [chunk.text for chunk in self.chunks]

    def save(self, index_dir: Path | str = INDEX_DIR) -> None:
        directory = Path(index_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / _CHUNKS_FILE).write_text(
            json.dumps([chunk.model_dump() for chunk in self.chunks], indent=2),
            encoding="utf-8",
        )
        (directory / _MANIFEST_FILE).write_text(
            json.dumps(self.manifest, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: Path | str = INDEX_DIR) -> "ChunkStore":
        directory = Path(index_dir)
        chunks_path = directory / _CHUNKS_FILE
        if not chunks_path.exists():
            return cls()

        chunks = [Chunk(**data) for data in json.loads(chunks_path.read_text(encoding="utf-8"))]
        manifest_path = directory / _MANIFEST_FILE
        manifest = (
            json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest_path.exists()
            else {}
        )
        return cls(chunks=chunks, manifest=manifest)
