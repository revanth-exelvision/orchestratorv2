from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage

from orchestrator.config import Settings
from orchestrator.main import create_app


@pytest.fixture
def fake_graph_output() -> dict[str, Any]:
    return {
        "plan": {
            "goal_summary": "Test goal",
            "steps": [
                {
                    "step_id": "1",
                    "description": "Do the thing",
                    "tool_name": "echo_text",
                    "inputs": "sample",
                    "expected_output": "echo",
                }
            ],
            "final_output_description": "A test result",
        },
        "messages": [AIMessage(content="Here is the final answer.")],
    }


@pytest.fixture
def mock_graph(monkeypatch: pytest.MonkeyPatch, fake_graph_output: dict[str, Any]):
    graph = AsyncMock()
    graph.ainvoke = AsyncMock(return_value=fake_graph_output)
    monkeypatch.setattr("orchestrator.main.GRAPH", graph)
    return graph


@pytest.fixture
def app(mock_graph):
    return create_app(settings=Settings.with_all_orchestration_routes())


@pytest.fixture
def client(app):
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c
