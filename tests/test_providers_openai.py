import httpx
import pytest

from app.errors import NonRetryableProviderError, ProviderError
from app.providers.openai_provider import OpenAIProvider
from app.tools.specs import get_openai_tool_specs


class _FakeMessage:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeChoice:
    def __init__(self, content=None, tool_calls=None):
        self.message = _FakeMessage(content, tool_calls)


class _FakeResponse:
    def __init__(self, content=None, tool_calls=None):
        self.choices = [_FakeChoice(content, tool_calls)]


class _FakeFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, name, arguments):
        self.function = _FakeFunction(name, arguments)


def _fake_request():
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


def _fake_http_response(status_code):
    return httpx.Response(status_code, request=_fake_request(), json={"error": {"message": "boom"}})


def _stub_create(monkeypatch, provider, *, result=None, tool_calls=None, exc=None):
    def fake_create(**kwargs):
        if exc is not None:
            raise exc
        return _FakeResponse(result, tool_calls)

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


def test_generate_tool_call_returns_name_and_parsed_arguments(monkeypatch):
    provider = OpenAIProvider(api_key="test-key")
    call = _FakeToolCall("get_project_status", '{"project_code": "PRJ-001"}')
    _stub_create(monkeypatch, provider, tool_calls=[call])

    result = provider.generate_tool_call(
        "What's the status of PRJ-001?", tools=get_openai_tool_specs()
    )

    assert result.name == "get_project_status"
    assert result.arguments == {"project_code": "PRJ-001"}


def test_generate_tool_call_returns_none_when_model_declines():
    provider = OpenAIProvider(api_key="test-key")

    def fake_create(**kwargs):
        return _FakeResponse(content="I can't help with that.", tool_calls=None)

    provider._client.chat.completions.create = fake_create

    result = provider.generate_tool_call("What's the weather today?", tools=get_openai_tool_specs())

    assert result is None


def test_generate_tool_call_raises_provider_error_on_malformed_arguments(monkeypatch):
    provider = OpenAIProvider(api_key="test-key")
    call = _FakeToolCall("get_project_status", "not valid json")
    _stub_create(monkeypatch, provider, tool_calls=[call])

    with pytest.raises(ProviderError, match="malformed"):
        provider.generate_tool_call("status of PRJ-001", tools=get_openai_tool_specs())


def test_generate_tool_call_wraps_timeout_as_retryable_provider_error(monkeypatch):
    from openai import APITimeoutError

    provider = OpenAIProvider(api_key="test-key")
    _stub_create(monkeypatch, provider, exc=APITimeoutError(request=_fake_request()))

    with pytest.raises(ProviderError):
        provider.generate_tool_call("hello", tools=get_openai_tool_specs())
