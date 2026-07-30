from typing import Protocol


class LLMProvider(Protocol):
    """Provider-neutral text-generation port every LLM adapter must satisfy.

    Kept intentionally minimal (single prompt/system in, string out) so the
    same protocol can back an intent classifier today and a RAG/answer
    generator later, regardless of which concrete model provider is behind
    it. Adapters must raise app.errors.ProviderError (or a subclass) on
    failure rather than leaking SDK-specific exceptions.
    """

    def generate(self, prompt: str, *, system: str | None = None) -> str: ...
