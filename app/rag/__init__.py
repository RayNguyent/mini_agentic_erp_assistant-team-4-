"""Project-document RAG: ingest, chunk, index, retrieve, ground, cite.

The pipeline's contract is that a document-route answer either cites a chunk id
that was actually retrieved, or refuses. That is enforced structurally in
app/rag/answer.py and again in AgentState, not by prompt wording.
"""

from app.rag.answer import GroundedAnswer, answer_from_documents, verify_citations
from app.rag.documents import Chunk, Document, ScoredChunk, load_corpus
from app.rag.embed import EmbeddingProvider, NullEmbeddingProvider, build_embedding_provider
from app.rag.retrieve import RetrievalResult, Retriever, build_retriever
from app.rag.store import ChunkStore
from app.rag.vector_index import (
    ChromaVectorIndex,
    NullVectorIndex,
    VectorIndex,
    build_vector_index,
)

__all__ = [
    "ChromaVectorIndex",
    "Chunk",
    "ChunkStore",
    "Document",
    "EmbeddingProvider",
    "GroundedAnswer",
    "NullEmbeddingProvider",
    "NullVectorIndex",
    "RetrievalResult",
    "Retriever",
    "ScoredChunk",
    "VectorIndex",
    "answer_from_documents",
    "build_embedding_provider",
    "build_retriever",
    "build_vector_index",
    "load_corpus",
    "verify_citations",
]
