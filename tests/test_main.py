from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app


def test_health(client: TestClient):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_orchestrate_json_success(client: TestClient, mock_graph, fake_graph_output):
    body = {
        "user_prompt": "Hello",
        "chat_history": [{"role": "user", "content": "Hi"}],
        "context": {"k": "v"},
    }
    r = client.post("/orchestrate/json", json=body)
    assert r.status_code == 200
    data = r.json()
    assert data["plan"]["goal_summary"] == fake_graph_output["plan"]["goal_summary"]
    assert data["answer"] == "Here is the final answer."
    mock_graph.ainvoke.assert_awaited_once()
    call_state = mock_graph.ainvoke.call_args[0][0]
    assert call_state["user_prompt"] == "Hello"
    assert call_state["chat_history"] == [{"role": "user", "content": "Hi"}]
    assert "Client context" in call_state["attachment_context"]


def test_orchestrate_plan_only(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from app.models import OrchestratorPlan

    async def fake_generate_plan(**kwargs: object) -> OrchestratorPlan:
        assert kwargs["user_prompt"] == "Plan me"
        assert "Client context" in kwargs["attachment_context"]
        return OrchestratorPlan(
            goal_summary="mock goal",
            steps=[],
            final_output_description="mock final",
        )

    monkeypatch.setattr("app.main.generate_plan", fake_generate_plan)
    r = client.post(
        "/orchestrate/plan",
        json={"user_prompt": "Plan me", "context": {"tenant": "t1"}},
    )
    assert r.status_code == 200
    data = r.json()
    assert data["goal_summary"] == "mock goal"
    assert data["final_output_description"] == "mock final"


def test_orchestrate_execute_only(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from langchain_core.messages import AIMessage

    async def fake_run_executor(**kwargs: object) -> list:
        assert kwargs["user_prompt"] == "Run it"
        assert kwargs["plan"]["goal_summary"] == "g"
        assert "Client context" in kwargs["attachment_context"]
        return [AIMessage(content="executor done")]

    monkeypatch.setattr("app.main.run_executor", fake_run_executor)
    r = client.post(
        "/orchestrate/execute",
        json={
            "plan": {
                "goal_summary": "g",
                "steps": [],
                "final_output_description": "f",
            },
            "user_prompt": "Run it",
            "context": {"x": 1},
        },
    )
    assert r.status_code == 200
    assert r.json() == {"answer": "executor done"}


def test_orchestrate_execute_only_validation_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    called: list[bool] = []

    async def fake_run_executor(**kwargs: object):
        called.append(True)
        raise AssertionError("should not be called")

    monkeypatch.setattr("app.main.run_executor", fake_run_executor)
    r = client.post("/orchestrate/execute", json={"user_prompt": "missing plan"})
    assert r.status_code == 422
    assert not called


def test_orchestrate_plan_only_validation_error(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    called: list[bool] = []

    async def fake_generate_plan(**kwargs: object):
        called.append(True)
        raise AssertionError("should not be called")

    monkeypatch.setattr("app.main.generate_plan", fake_generate_plan)
    r = client.post("/orchestrate/plan", json={})
    assert r.status_code == 422
    assert not called


def test_orchestrate_json_validation_error(client: TestClient, mock_graph):
    r = client.post("/orchestrate/json", json={})
    assert r.status_code == 422
    mock_graph.ainvoke.assert_not_awaited()


def test_orchestrate_multipart_success(client: TestClient, mock_graph):
    payload = json.dumps({"user_prompt": "From multipart"})
    r = client.post(
        "/orchestrate",
        data={"payload": payload},
    )
    assert r.status_code == 200
    assert r.json()["answer"] == "Here is the final answer."


def test_orchestrate_multipart_invalid_json(client: TestClient, mock_graph):
    r = client.post("/orchestrate", data={"payload": "not-json{"})
    assert r.status_code == 400
    mock_graph.ainvoke.assert_not_awaited()


def test_orchestrate_multipart_invalid_payload_shape(client: TestClient, mock_graph):
    r = client.post("/orchestrate", data={"payload": json.dumps({"chat_history": []})})
    assert r.status_code == 400
    mock_graph.ainvoke.assert_not_awaited()


def test_orchestrate_multipart_content_length_too_large(client: TestClient, mock_graph):
    from app.config import Settings, get_settings

    app.dependency_overrides[get_settings] = lambda: Settings(max_total_request_bytes=100)
    try:
        r = client.post(
            "/orchestrate",
            data={"payload": json.dumps({"user_prompt": "x"})},
            headers={"content-length": "999999"},
        )
        assert r.status_code == 413
        mock_graph.ainvoke.assert_not_awaited()
    finally:
        app.dependency_overrides.clear()


def test_list_orchestrate_flows(client: TestClient):
    r = client.get("/orchestrate/flows")
    assert r.status_code == 200
    data = r.json()
    assert isinstance(data, list)
    ids = [item["flow_id"] for item in data]
    assert ids == sorted(ids)
    assert "echo_smoke" in ids
    assert "word_stats" in ids
    assert "agent_metrics" in ids
    assert "agent_reverse" in ids
    assert "agent_bullets" in ids
    for item in data:
        assert "title" in item and "description" in item


def test_orchestrate_named_flow_unknown(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    async def fake_run_executor(**kwargs: object):
        raise AssertionError("should not be called")

    monkeypatch.setattr("app.main.run_executor", fake_run_executor)
    r = client.post(
        "/orchestrate/flows/does_not_exist",
        json={"user_prompt": "hello"},
    )
    assert r.status_code == 404


def test_orchestrate_named_flow_success(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    from langchain_core.messages import AIMessage

    captured: dict[str, object] = {}

    async def fake_run_executor(**kwargs: object) -> list:
        captured.update(kwargs)
        return [AIMessage(content="named flow done")]

    monkeypatch.setattr("app.main.run_executor", fake_run_executor)
    r = client.post(
        "/orchestrate/flows/word_stats",
        json={
            "user_prompt": "Count these words",
            "chat_history": [{"role": "user", "content": "prior"}],
            "context": {"c": 1},
        },
    )
    assert r.status_code == 200
    assert r.json() == {"answer": "named flow done"}
    assert captured["user_prompt"] == "Count these words"
    assert captured["chat_history"] == [{"role": "user", "content": "prior"}]
    plan = captured["plan"]
    assert isinstance(plan, dict)
    assert plan["goal_summary"] == "Report how many words are in the user's text"
    assert "Client context" in str(captured.get("attachment_context", ""))
