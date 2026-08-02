from app.tools.specs import get_openai_tool_specs


def _spec_by_name(specs, name):
    return next(s for s in specs if s["function"]["name"] == name)


def test_returns_one_spec_per_tool():
    specs = get_openai_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {
        "get_project_status",
        "list_risks",
        "create_risk",
        "list_project_tasks",
        "get_sprint_progress",
        "get_budget_summary",
    }


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


def test_create_risk_offers_project_code_and_risk_payload():
    fn = _spec_by_name(get_openai_tool_specs(), "create_risk")["function"]
    assert {"project_code", "risk_payload"} <= set(fn["parameters"]["properties"])


def test_create_risk_requires_nothing_so_a_bare_request_still_calls_it():
    """An under-specified "add a risk" must still produce a tool call — that
    is what routes the user to the approval form instead of a chat reply
    asking for the details."""
    fn = _spec_by_name(get_openai_tool_specs(), "create_risk")["function"]
    assert not fn["parameters"].get("required")


# --- MCP-style tool boundary metadata -----------------------------------------


def test_every_tool_has_boundary_metadata():
    from app.tools.specs import TOOL_META, get_openai_tool_specs

    spec_names = {s["function"]["name"] for s in get_openai_tool_specs()}
    assert spec_names <= set(TOOL_META)


def test_only_create_risk_is_a_write_tool():
    from app.tools.specs import TOOL_META

    writes = {name for name, meta in TOOL_META.items() if meta.side_effect == "write"}
    assert writes == {"create_risk"}


def test_budget_summary_requires_the_finance_permission_not_project_read():
    from app.tools.specs import permission_for

    assert permission_for("get_budget_summary") == "project.finance.read"
    assert permission_for("get_project_status") == "project.read"


def test_a_write_tool_is_never_blindly_retried():
    from app.tools.specs import TOOL_META

    assert TOOL_META["create_risk"].retry_limit == 0
