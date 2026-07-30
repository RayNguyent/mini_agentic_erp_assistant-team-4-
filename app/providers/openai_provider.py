import json
import logging
import time

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from app.errors import NonRetryableProviderError, ProviderError
from app.providers.base import ToolCall
from app.tools.specs import ToolSpec

logger = logging.getLogger(__name__)


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

    def generate(
        self, prompt: str, *, system: str | None = None, history: list[dict] | None = None
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        logger.info(
            "openai request: mode=generate model=%s base_url=%s prompt_chars=%d",
            self._model, self._client.base_url, len(prompt),
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
            )
        except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
            logger.warning(
                "openai request failed: mode=generate model=%s elapsed_ms=%d error=%s",
                self._model, (time.monotonic() - start) * 1000, exc,
            )
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        except (AuthenticationError, BadRequestError) as exc:
            logger.warning(
                "openai request rejected: mode=generate model=%s elapsed_ms=%d error=%s",
                self._model, (time.monotonic() - start) * 1000, exc,
            )
            raise NonRetryableProviderError(f"OpenAI request rejected: {exc}") from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        content = response.choices[0].message.content
        logger.info(
            "openai response: mode=generate model=%s elapsed_ms=%d response_chars=%d",
            self._model, elapsed_ms, len(content or ""),
        )
        if not content:
            raise ProviderError("OpenAI returned an empty response")
        return content

    def generate_text(
        self, prompt: str, *, system: str | None = None, history: list[dict] | None = None
    ) -> str:
        """Generate plain text response without JSON mode constraint."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        logger.info(
            "openai request: mode=generate_text model=%s base_url=%s prompt_chars=%d",
            self._model, self._client.base_url, len(prompt),
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )
        except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
            logger.warning(
                "openai request failed: mode=generate_text model=%s elapsed_ms=%d error=%s",
                self._model, (time.monotonic() - start) * 1000, exc,
            )
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        except (AuthenticationError, BadRequestError) as exc:
            logger.warning(
                "openai request rejected: mode=generate_text model=%s elapsed_ms=%d error=%s",
                self._model, (time.monotonic() - start) * 1000, exc,
            )
            raise NonRetryableProviderError(f"OpenAI request rejected: {exc}") from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        content = response.choices[0].message.content
        logger.info(
            "openai response: mode=generate_text model=%s elapsed_ms=%d response_chars=%d",
            self._model, elapsed_ms, len(content or ""),
        )
        if not content:
            raise ProviderError("OpenAI returned an empty response")
        return content

    def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: list[ToolSpec],
        system: str | None = None,
        history: list[dict] | None = None,
    ) -> ToolCall | None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": prompt})

        start = time.monotonic()
        logger.info(
            "openai request: mode=tool_call model=%s base_url=%s prompt_chars=%d tools=%d",
            self._model, self._client.base_url, len(prompt), len(tools),
        )
        logger.debug("openai tool-call system_prompt=%s user_prompt=%s", system, prompt)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
        except (APITimeoutError, RateLimitError, APIConnectionError) as exc:
            logger.warning(
                "openai request failed: mode=tool_call model=%s elapsed_ms=%d error=%s",
                self._model, (time.monotonic() - start) * 1000, exc,
            )
            raise ProviderError(f"OpenAI request failed: {exc}") from exc
        except (AuthenticationError, BadRequestError) as exc:
            logger.warning(
                "openai request rejected: mode=tool_call model=%s elapsed_ms=%d error=%s",
                self._model, (time.monotonic() - start) * 1000, exc,
            )
            raise NonRetryableProviderError(f"OpenAI request rejected: {exc}") from exc

        elapsed_ms = (time.monotonic() - start) * 1000
        calls = response.choices[0].message.tool_calls
        logger.info(
            "openai response: mode=tool_call model=%s elapsed_ms=%d tool_call=%s arguments=%s",
            self._model, elapsed_ms,
            calls[0].function.name if calls else None,
            calls[0].function.arguments if calls else None,
        )
        if not calls:
            return None

        call = calls[0]
        try:
            arguments = json.loads(call.function.arguments)
        except json.JSONDecodeError as exc:
            logger.warning(
                "openai tool-call arguments malformed: model=%s name=%s error=%s",
                self._model, call.function.name, exc,
            )
            raise ProviderError(
                f"OpenAI returned malformed tool-call arguments: {exc}"
            ) from exc

        return ToolCall(name=call.function.name, arguments=arguments)
