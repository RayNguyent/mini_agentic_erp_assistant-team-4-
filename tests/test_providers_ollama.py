"""OllamaProvider: it's an OpenAIProvider with Ollama-shaped defaults, so this
only tests the defaulting, not the request/response handling (already covered
by tests/test_providers_openai.py, which OllamaProvider inherits behavior
from)."""

from app.providers.config import OLLAMA_DEFAULT_BASE_URL
from app.providers.ollama_provider import OllamaProvider


def test_defaults_to_the_local_ollama_endpoint():
    provider = OllamaProvider()
    assert str(provider._client.base_url).rstrip("/") == OLLAMA_DEFAULT_BASE_URL.rstrip("/")


def test_an_explicit_base_url_overrides_the_default():
    provider = OllamaProvider(base_url="http://remote-ollama:11434/v1")
    assert "remote-ollama" in str(provider._client.base_url)


def test_requires_no_real_credential():
    # Must not raise for lacking a real API key — Ollama ignores it.
    OllamaProvider()


def test_default_model_is_a_locally_plausible_name():
    provider = OllamaProvider()
    assert provider._model == "llama3.1"


def test_is_an_openai_provider_and_inherits_its_behavior():
    from app.providers.openai_provider import OpenAIProvider

    assert isinstance(OllamaProvider(), OpenAIProvider)
