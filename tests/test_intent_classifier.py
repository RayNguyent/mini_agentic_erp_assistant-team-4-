from app.errors import NonRetryableProviderError, ProviderError
from app.providers.base import ToolCall
from app.providers.intent_classifier import make_llm_classifier, make_tool_calling_classifier
from app.runtime import default_classify


class FakeProvider:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def generate(self, prompt, *, system=None, history=None):
        self.calls.append((prompt, system, history))
        if self._exc is not None:
            raise self._exc
        return self._response


class FakeToolCallProvider:
    def __init__(self, call=None, exc=None):
        self._call = call
        self._exc = exc
        self.calls = []

    def generate_tool_call(self, prompt, *, tools, system=None, history=None):
        self.calls.append((prompt, tools, system, history))
        if self._exc is not None:
            raise self._exc
        return self._call


def test_classifier_parses_valid_llm_response():
    provider = FakeProvider(response='{"intent": "project_status", "project_code": "PRJ-001"}')
    classify = make_llm_classifier(provider)

    intent, tool_input = classify("What's the status of PRJ-001?")

    assert intent == "project_status"
    assert tool_input == {"project_code": "PRJ-001"}
    assert len(provider.calls) == 1


def test_classifier_omits_project_code_when_null():
    provider = FakeProvider(response='{"intent": "unsupported", "project_code": null}')
    classify = make_llm_classifier(provider)

    intent, tool_input = classify("What's the weather today?")

    assert intent == "unsupported"
    assert tool_input == {}


def test_classifier_falls_back_on_retryable_provider_error():
    message = "Create a risk for PRJ-001"
    provider = FakeProvider(exc=ProviderError("upstream down"))
    classify = make_llm_classifier(provider)

    assert classify(message) == default_classify(message)


def test_classifier_falls_back_on_non_retryable_provider_error():
    message = "Create a risk for PRJ-001"
    provider = FakeProvider(exc=NonRetryableProviderError("bad credentials"))
    classify = make_llm_classifier(provider)

    assert classify(message) == default_classify(message)


def test_classifier_falls_back_on_malformed_json():
    message = "Show all risks for PRJ-001"
    provider = FakeProvider(response="not json at all")
    classify = make_llm_classifier(provider)

    assert classify(message) == default_classify(message)


def test_classifier_falls_back_on_invalid_intent_value():
    message = "status of PRJ-001"
    provider = FakeProvider(response='{"intent": "not_a_real_intent"}')
    classify = make_llm_classifier(provider)

    assert classify(message) == default_classify(message)


def test_classifier_falls_back_to_custom_fallback_when_provided():
    calls = []

    def custom_fallback(message, history=None):
        calls.append(message)
        return "unsupported", {}

    provider = FakeProvider(exc=ProviderError("down"))
    classify = make_llm_classifier(provider, fallback=custom_fallback)

    result = classify("anything")

    assert result == ("unsupported", {})
    assert calls == ["anything"]


def test_llm_classifier_forwards_history_to_provider():
    provider = FakeProvider(response='{"intent": "project_status", "project_code": "PRJ-003"}')
    classify = make_llm_classifier(provider)
    history = [{"role": "user", "content": "What's the status of PRJ-001?"}]

    classify("how about PRJ-003?", history)

    assert provider.calls[0][2] == history


def test_tool_calling_classifier_maps_tool_name_to_intent():
    call = ToolCall(name="get_project_status", arguments={"project_code": "PRJ-001"})
    provider = FakeToolCallProvider(call=call)
    classify = make_tool_calling_classifier(provider)

    intent, tool_input = classify("What's the status of PRJ-001?")

    assert intent == "project_status"
    assert tool_input == {"project_code": "PRJ-001"}


def test_tool_calling_classifier_passes_through_full_create_risk_payload():
    call = ToolCall(
        name="create_risk",
        arguments={
            "project_code": "PRJ-001",
            "risk_payload": {"title": "Scope creep", "severity": "medium", "description": ""},
        },
    )
    provider = FakeToolCallProvider(call=call)
    classify = make_tool_calling_classifier(provider)

    intent, tool_input = classify("Create a risk for PRJ-001 about scope creep, medium severity")

    assert intent == "create_risk"
    assert tool_input["risk_payload"]["title"] == "Scope creep"


def test_tool_calling_classifier_returns_unsupported_when_model_declines():
    provider = FakeToolCallProvider(call=None)
    classify = make_tool_calling_classifier(provider)

    intent, tool_input = classify("What's the weather today?")

    assert intent == "unsupported"
    assert tool_input == {}


def test_tool_calling_classifier_falls_back_on_provider_error():
    message = "Create a risk for PRJ-001"
    provider = FakeToolCallProvider(exc=ProviderError("upstream down"))
    classify = make_tool_calling_classifier(provider)

    assert classify(message) == default_classify(message)


def test_tool_calling_classifier_forwards_history_to_provider():
    call = ToolCall(name="get_project_status", arguments={"project_code": "PRJ-003"})
    provider = FakeToolCallProvider(call=call)
    classify = make_tool_calling_classifier(provider)
    history = [{"role": "user", "content": "What's the status of PRJ-001?"}]

    classify("how about PRJ-003?", history)

    assert provider.calls[0][3] == history


def test_tool_calling_classifier_falls_back_on_unknown_tool_name():
    call = ToolCall(name="delete_everything", arguments={})
    provider = FakeToolCallProvider(call=call)
    message = "status of PRJ-001"
    classify = make_tool_calling_classifier(provider)

    assert classify(message) == default_classify(message)
