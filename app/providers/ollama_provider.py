"""Ollama adapter: OpenAIProvider pointed at Ollama's OpenAI-compatible
endpoint.

Ollama exposes chat completions (including JSON mode and tool calling) under
`/v1`, so no separate client or response-parsing code is needed — the only
adapter-specific behavior is the default base URL and that no real API key is
required (Ollama ignores it, so a placeholder is supplied rather than making
the caller invent one).
"""

from app.providers.config import OLLAMA_DEFAULT_BASE_URL
from app.providers.openai_provider import OpenAIProvider


class OllamaProvider(OpenAIProvider):
    def __init__(
        self,
        *,
        model: str = "llama3.1",
        base_url: str | None = None,
        timeout_s: float = 30.0,  # local inference is slower than a hosted API
    ) -> None:
        super().__init__(
            api_key="ollama",  # ignored by Ollama; OpenAI's client requires a non-empty string
            model=model,
            base_url=base_url or OLLAMA_DEFAULT_BASE_URL,
            timeout_s=timeout_s,
        )
