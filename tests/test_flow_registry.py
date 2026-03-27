from __future__ import annotations

from orchestrator.flow_registry import DEFAULT_FLOWS, get_flow, list_flow_summaries
from orchestrator.models import OrchestratorPlan


def test_list_flow_summaries_default_sorted_by_flow_id():
    rows = list_flow_summaries()
    ids = [r.flow_id for r in rows]
    assert ids == sorted(ids)
    assert len(ids) == len(DEFAULT_FLOWS)
    by_id = {r.flow_id: r for r in rows}
    assert by_id["word_stats"].title == "Word statistics"


def test_list_flow_summaries_explicit_registry():
    custom = {
        "z_last": ("Z", "d", OrchestratorPlan(goal_summary="z", steps=[], final_output_description="")),
        "a_first": ("A", "d", OrchestratorPlan(goal_summary="a", steps=[], final_output_description="")),
    }
    rows = list_flow_summaries(custom)
    assert [r.flow_id for r in rows] == ["a_first", "z_last"]


def test_get_flow_default_registry():
    plan = get_flow("echo_smoke")
    assert plan is not None
    assert plan.goal_summary == "Echo user-provided text back"


def test_get_flow_explicit_registry():
    only = OrchestratorPlan(goal_summary="mine", steps=[], final_output_description="x")
    reg = {"x": ("t", "d", only)}
    assert get_flow("x", reg) == only
    assert get_flow("missing", reg) is None


def test_get_flow_unknown_returns_none():
    assert get_flow("not_a_real_flow_id_ever") is None
