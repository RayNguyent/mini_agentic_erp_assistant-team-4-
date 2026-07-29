from app.tools.specs import get_openai_tool_specs


def _spec_by_name(specs, name):
    return next(s for s in specs if s["function"]["name"] == name)


def test_returns_one_spec_per_tool():
    specs = get_openai_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {"get_project_status", "list_risks", "create_risk"}


def test_every_spec_is_openai_function_shape():
    for spec in get_openai_tool_specs():
        assert spec["type"] == "function"
        fn = spec["function"]
        assert isinstance(fn["name"], str) and fn["name"]
        assert isinstance(fn["description"], str) and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_get_project_status_requires_project_code():
    fn = _spec_by_name(get_openai_tool_specs(), "get_project_status")["function"]
    assert fn["parameters"]["required"] == ["project_code"]


def test_list_risks_requires_only_project_code():
    fn = _spec_by_name(get_openai_tool_specs(), "list_risks")["function"]
    assert fn["parameters"]["required"] == ["project_code"]
    assert "risk_payload" not in fn["parameters"]["properties"]


def test_create_risk_requires_project_code_and_risk_payload():
    fn = _spec_by_name(get_openai_tool_specs(), "create_risk")["function"]
    assert set(fn["parameters"]["required"]) == {"project_code", "risk_payload"}
    assert "risk_payload" in fn["parameters"]["properties"]
