import os

from pydantic import BaseModel


class ProviderSettings(BaseModel):
    """Env-driven configuration for which LLM provider backs the runtime.

    Deterministic mode is the default and requires no credential, so the
    app stays demoable offline; switching to "openai" additionally requires
    OPENAI_API_KEY.
    """

    llm_provider: str = "deterministic"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str | None = None
    openai_timeout_s: float = 10.0

    @classmethod
    def from_env(cls) -> "ProviderSettings":
        return cls(
            llm_provider=os.environ.get("LLM_PROVIDER", "deterministic"),
            openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
            openai_model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            openai_base_url=os.environ.get("OPENAI_BASE_URL") or None,
            openai_timeout_s=float(os.environ.get("OPENAI_TIMEOUT_S", "10")),
        )
