"""Okapi BM25, implemented from scratch in pure Python.

This is the offline guarantee for the whole RAG layer. Embeddings require a
configured provider and a credential; BM25 requires neither, so the citation
contract still holds when the deterministic profile runs with no network. When
embeddings *are* available the two are fused (app/rag/retrieve.py) rather than
one replacing the other — keyword matching remains the stronger signal for the
exact identifiers this corpus is full of (PRJ-001, RISK-2, 429, USD figures).
"""

import math
import re
from collections import Counter

K1 = 1.5  # term-frequency saturation
B = 0.75  # length normalisation

# Small, deliberately conservative list. Domain terms (risk, budget, sprint,
# open, high) are NOT stopwords here — they carry real meaning in this corpus.
_STOPWORDS = frozenset(
    """
    a an the and or but if then else of for to in on at by with without from as
    is are was were be been being do does did doing have has had having it its
    this that these those there here what which who whom whose when where why how
    i you he she we they me him her us them my your our their can could should
    would may might must will shall about into over under again further once
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-_][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    """Lowercase word/identifier tokens, stopwords removed.

    Identifiers keep their internal separators (`prj-001` stays one token) so an
    exact project code is a precise match rather than two common fragments.
    """
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


class BM25Index:
    """An in-memory inverted index over a fixed document list."""

    def __init__(self, documents: list[str]) -> None:
        self._docs_tokens = [tokenize(doc) for doc in documents]
        self._doc_count = len(self._docs_tokens)
        self._doc_lengths = [len(tokens) for tokens in self._docs_tokens]
        self._avg_length = (
            sum(self._doc_lengths) / self._doc_count if self._doc_count else 0.0
        )

        self._term_freqs: list[Counter[str]] = [Counter(t) for t in self._docs_tokens]
        doc_freq: Counter[str] = Counter()
        for tokens in self._docs_tokens:
            doc_freq.update(set(tokens))
        self._doc_freq = doc_freq

        # Precomputed so scoring is a lookup, not a log() per term per document.
        self._idf = {
            term: math.log(1 + (self._doc_count - freq + 0.5) / (freq + 0.5))
            for term, freq in doc_freq.items()
        }

    @property
    def size(self) -> int:
        return self._doc_count

    @property
    def vocabulary_size(self) -> int:
        return len(self._doc_freq)

    def score(self, query: str) -> list[float]:
        query_terms = tokenize(query)
        scores = [0.0] * self._doc_count
        if not query_terms or not self._doc_count:
            return scores

        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for index in range(self._doc_count):
                freq = self._term_freqs[index].get(term)
                if not freq:
                    continue
                length_norm = 1 - B + B * (
                    self._doc_lengths[index] / self._avg_length if self._avg_length else 1
                )
                scores[index] += idf * (freq * (K1 + 1)) / (freq + K1 * length_norm)

        return scores

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return `(document_index, score)` for the best `top_k` matches.

        Zero-scoring documents are dropped rather than padded out — an empty
        result is a real answer and must reach the refusal path intact.
        """
        scored = [(i, s) for i, s in enumerate(self.score(query)) if s > 0]
        scored.sort(key=lambda pair: (-pair[1], pair[0]))
        return scored[:top_k]
