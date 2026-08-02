"""Embedding provider port and adapters.

Same shape as app/providers/base.py: a Protocol the retriever depends on, with
concrete adapters kept behind it. The retriever must work when `available` is
False — that is the whole reason this is a port rather than a direct SDK call.
"""

import logging
import time
from typing import Protocol, runtime_checkable

from app.errors import NonRetryableProviderError, ProviderError

logger = logging.getLogger(__name__)


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turns text into vectors. Adapters raise app.errors.ProviderError (or
    NonRetryableProviderError) — never a vendor SDK exception."""

    name: str
    dimensions: int
    available: bool

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


class NullEmbeddingProvider:
    """The deterministic offline default: no embeddings at all.

    Returns nothing rather than fake vectors. A zero vector would silently rank
    every chunk equally and make a broken index look like a working one; an
    explicit `available = False` sends the retriever down the BM25-only path
    where the degradation is visible in the diagnostics and in /readiness.
    """

    name = "none"
    dimensions = 0
    available = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return []

    def embed_query(self, text: str) -> list[float]:
        return []


class OpenAIEmbeddingAdapter:
    """OpenAI-compatible embeddings. `base_url` also covers Ollama and any
    other service exposing the /v1/embeddings shape."""

    available = True

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str | None = None,
        timeout_s: float = 20.0,
        dimensions: int = 1536,
    ) -> None:
        from openai import OpenAI

        self.name = f"openai:{model}"
        self.model = model
        self.dimensions = dimensions
        self._client = OpenAI(
            api_key=api_key, base_url=base_url or None, timeout=timeout_s
        )

    def _embed(self, texts: list[str]) -> list[list[float]]:
        from openai import (
            APIConnectionError,
            APITimeoutError,
            AuthenticationError,
            BadRequestError,
            RateLimitError,
        )

        if not texts:
            return []

        started = time.perf_counter()
        try:
            response = self._client.embeddings.create(model=self.model, input=texts)
        except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
            raise ProviderError(f"embedding provider unavailable: {exc}") from exc
        except (AuthenticationError, BadRequestError) as exc:
            raise NonRetryableProviderError(f"embedding request rejected: {exc}") from exc

        logger.info(
            "embeddings model=%s count=%d elapsed_ms=%.1f",
            self.model,
            len(texts),
            (time.perf_counter() - started) * 1000,
        )
        return [item.embedding for item in response.data]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Batched to stay under per-request input limits on large corpora.
        vectors: list[list[float]] = []
        for start in range(0, len(texts), 64):
            vectors.extend(self._embed(texts[start : start + 64]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text])
        return vectors[0] if vectors else []


def build_embedding_provider(settings=None) -> EmbeddingProvider:
    """Pick an adapter from environment configuration, degrading to Null.

    Never raises on a missing credential: an unconfigured embedding provider is
    a supported operating mode, not an error.
    """
    from app.providers.config import ProviderSettings

    settings = settings or ProviderSettings.from_env()
    if not settings.embeddings_enabled:
        return NullEmbeddingProvider()

    try:
        return OpenAIEmbeddingAdapter(
            api_key=settings.openai_api_key or "not-needed",
            model=settings.embedding_model,
            base_url=settings.openai_base_url,
            timeout_s=settings.openai_timeout_s,
        )
    except Exception as exc:  # noqa: BLE001 - degrade, never fail startup
        logger.warning("embedding provider unavailable, using BM25 only: %s", exc)
        return NullEmbeddingProvider()
