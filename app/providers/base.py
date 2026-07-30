from typing import Protocol

from pydantic import BaseModel

from app.tools.specs import ToolSpec


class ToolCall(BaseModel):
    name: str
    arguments: dict


class LLMProvider(Protocol):
    """Provider-neutral text-generation port every LLM adapter must satisfy.

    Kept intentionally minimal (single prompt/system in, string out) so the
    same protocol can back an intent classifier today and a RAG/answer
    generator later, regardless of which concrete model provider is behind
    it. Adapters must raise app.errors.ProviderError (or a subclass) on
    failure rather than leaking SDK-specific exceptions.
    """

    def generate(
        self, prompt: str, *, system: str | None = None, history: list[dict] | None = None
    ) -> str: ...

    def generate_tool_call(
        self,
        prompt: str,
        *,
        tools: list[ToolSpec],
        system: str | None = None,
        history: list[dict] | None = None,
    ) -> ToolCall | None:
        """Ask the model to pick one of `tools` and return its name + parsed
        arguments, or None if the model declines to call any tool."""
        ...
