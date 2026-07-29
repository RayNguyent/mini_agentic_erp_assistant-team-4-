from app.errors import NonRetryableProviderError, ProviderError
from app.providers.intent_classifier import make_llm_classifier
from app.runtime import default_classify


class FakeProvider:
    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc
        self.calls = []

    def generate(self, prompt, *, system=None):
        self.calls.append((prompt, system))
        if self._exc is not None:
            raise self._exc
        return self._response


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

    def custom_fallback(message):
        calls.append(message)
        return "unsupported", {}

    provider = FakeProvider(exc=ProviderError("down"))
    classify = make_llm_classifier(provider, fallback=custom_fallback)

    result = classify("anything")

    assert result == ("unsupported", {})
    assert calls == ["anything"]
