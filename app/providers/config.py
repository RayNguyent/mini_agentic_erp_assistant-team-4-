import os

from pydantic import BaseModel

# Provider modes. "ollama" is an OpenAI-compatible endpoint with a different
# default base URL and no real credential requirement.
DETERMINISTIC = "deterministic"
OPENAI = "openai"
OLLAMA = "ollama"

OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"


def _flag(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


class ProviderSettings(BaseModel):
    """Env-driven configuration for which providers back the runtime.

    Deterministic mode is the default and requires no credential, so the app
    stays demoable offline; switching to "openai" additionally requires
    OPENAI_API_KEY, and "ollama" requires a locally running server.
    """

    llm_provider: str = DETERMINISTIC
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    openai_timeout_s: float = 10.0

    # Embeddings are a separate switch from generation: a local Ollama can
    # serve embeddings while generation stays deterministic, and BM25-only
    # retrieval remains a fully supported mode rather than a failure.
    embeddings_enabled: bool = False
    embedding_model: str = "text-embedding-3-small"

    # Tool + document configuration.
    tool_timeout_s: float = 10.0
    erp_provider: str = "mock"
    index_dir: str = "data/index"
    corpus_dir: str = "docs/corpus"

    @property
    def is_live(self) -> bool:
        """True when generation goes to a real provider rather than the
        deterministic offline path."""
        if self.llm_provider == OPENAI:
            return bool(self.openai_api_key)
        return self.llm_provider == OLLAMA

    @property
    def resolved_base_url(self) -> str | None:
        if self.openai_base_url:
            return self.openai_base_url
        return OLLAMA_DEFAULT_BASE_URL if self.llm_provider == OLLAMA else None

    def redacted(self) -> dict:
        """Safe for /readiness, traces, and screenshots — the credential is
        reported as present/absent, never echoed."""
        return {
            "llm_provider": self.llm_provider,
            "model": self.openai_model,
            "base_url": self.resolved_base_url,
            "timeout_s": self.openai_timeout_s,
            "credential_configured": bool(self.openai_api_key),
            "embeddings_enabled": self.embeddings_enabled,
            "embedding_model": self.embedding_model if self.embeddings_enabled else None,
            "erp_provider": self.erp_provider,
        }

    @classmethod
    def from_env(cls) -> "ProviderSettings":
        return cls(
            llm_provider=os.environ.get("LLM_PROVIDER", DETERMINISTIC),
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
            openai_timeout_s=float(os.environ.get("OPENAI_TIMEOUT_S", "10")),
            embeddings_enabled=_flag("EMBEDDINGS_ENABLED", False),
            embedding_model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
            tool_timeout_s=float(os.environ.get("TOOL_TIMEOUT_S", "10")),
            erp_provider=os.environ.get("ERP_PROVIDER", "mock"),
            index_dir=os.environ.get("INDEX_DIR", "data/index"),
            corpus_dir=os.environ.get("CORPUS_DIR", "docs/corpus"),
        )
