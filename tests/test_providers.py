from app.providers.erp import MockERPProvider


def test_get_project_returns_known_project():
    provider = MockERPProvider()
    project = provider.get_project("PRJ-001")
    assert project is not None
    assert project["project_code"] == "PRJ-001"


def test_get_project_returns_none_for_unknown_code():
    provider = MockERPProvider()
    assert provider.get_project("PRJ-999") is None


def test_list_risks_filters_by_project_code():
    provider = MockERPProvider()
    risks = provider.list_risks("PRJ-001")
    assert len(risks) == 2
    assert all(risk["project_code"] == "PRJ-001" for risk in risks)


def test_list_risks_empty_for_project_without_risks():
    provider = MockERPProvider()
    assert provider.list_risks("PRJ-002") == []


def test_create_risk_appends_and_returns_new_risk():
    provider = MockERPProvider()
    before = len(provider.list_risks("PRJ-002"))

    risk = provider.create_risk(
        "PRJ-002",
        {"title": "New vendor risk", "severity": "high", "description": "Vendor slipped milestone."},
    )

    assert risk["project_code"] == "PRJ-002"
    assert risk["status"] == "Open"
    assert risk["id"]
    assert len(provider.list_risks("PRJ-002")) == before + 1


def test_create_risk_ids_are_unique_and_increasing():
    provider = MockERPProvider()
    first = provider.create_risk("PRJ-002", {"title": "A", "severity": "low", "description": ""})
    second = provider.create_risk("PRJ-002", {"title": "B", "severity": "low", "description": ""})
    assert first["id"] != second["id"]
