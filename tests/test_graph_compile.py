from __future__ import annotations

from orchestrator.graph import GRAPH, build_compiled_graph


def test_build_compiled_graph_returns_runnable():
    g = build_compiled_graph()
    assert hasattr(g, "ainvoke")


def test_module_graph_is_compiled():
    assert hasattr(GRAPH, "ainvoke")
