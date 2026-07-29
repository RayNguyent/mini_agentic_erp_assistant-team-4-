import httpx
import pytest

from app.errors import NonRetryableProviderError, ProviderError
from app.providers.openai_provider import OpenAIProvider


class _FakeMessage:
    def __init__(self, content):
        self.content = content


class _FakeChoice:
    def __init__(self, content):
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


def _fake_request():
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _fake_http_response(status_code):
    return httpx.Response(status_code, request=_fake_request(), json={"error": {"message": "boom"}})


def _stub_create(monkeypatch, provider, *, result=None, exc=None):
    def fake_create(**kwargs):
        if exc is not None:
            raise exc
        return _FakeResponse(result)

    monkeypatch.setattr(provider._client.chat.completions, "create", fake_create)


def test_generate_returns_content_on_success(monkeypatch):
    provider = OpenAIProvider(api_key="test-key")
    _stub_create(monkeypatch, provider, result='{"intent": "project_status"}')

    result = provider.generate("What's the status of PRJ-001?", system="classify")

    assert result == '{"intent": "project_status"}'


def test_generate_wraps_timeout_as_retryable_provider_error(monkeypatch):
    from openai import APITimeoutError

    provider = OpenAIProvider(api_key="test-key")
    _stub_create(monkeypatch, provider, exc=APITimeoutError(request=_fake_request()))

    with pytest.raises(ProviderError):
        provider.generate("hello")


def test_generate_wraps_connection_error_as_retryable_provider_error(monkeypatch):
    from openai import APIConnectionError

    provider = OpenAIProvider(api_key="test-key")
    _stub_create(monkeypatch, provider, exc=APIConnectionError(request=_fake_request()))

    with pytest.raises(ProviderError):
        provider.generate("hello")


def test_generate_wraps_authentication_error_as_non_retryable(monkeypatch):
    from openai import AuthenticationError

    provider = OpenAIProvider(api_key="test-key")
    err = AuthenticationError("invalid api key", response=_fake_http_response(401), body=None)
    _stub_create(monkeypatch, provider, exc=err)

    with pytest.raises(NonRetryableProviderError):
        provider.generate("hello")


def test_generate_raises_provider_error_on_empty_content(monkeypatch):
    provider = OpenAIProvider(api_key="test-key")
    _stub_create(monkeypatch, provider, result=None)

    with pytest.raises(ProviderError, match="empty"):
        provider.generate("hello")
