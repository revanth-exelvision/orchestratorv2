from __future__ import annotations

import orchestrator


def test_all_public_names_are_defined():
    for name in orchestrator.__all__:
        assert hasattr(orchestrator, name), f"missing export: {name}"


def test_default_tools_alias_matches():
    assert orchestrator.TOOLS is orchestrator.DEFAULT_TOOLS
