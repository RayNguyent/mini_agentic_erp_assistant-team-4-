from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from app.errors import NonRetryableProviderError, ProviderError


class OpenAIProvider:
    """LLMProvider backed by the OpenAI chat completions API.

    Uses chat-completions JSON mode (response_format={"type": "json_object"})
    rather than OpenAI's schema-only Structured Outputs, since JSON mode is
    also supported by OpenAI-compatible endpoints (e.g. Ollama), keeping this
    adapter's shape reusable for a future compatible provider.
    """

    def __init__(
        self,
        api_key: str,
        *,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        timeout_s: float = 10.0,
    ):
        self._model = model
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout_s)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        except (AuthenticationError, BadRequestError) as exc:
            raise NonRetryableProviderError(f"OpenAI request rejected: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise ProviderError("OpenAI returned an empty response")
        return content
