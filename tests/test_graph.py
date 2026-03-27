from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.graph import (
    _build_executor_messages,
    _history_lines,
    _tools_manifest,
    last_assistant_text,
)
from app.tools import TOOLS


def test_last_assistant_text_prefers_latest_ai_string():
    msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="first"),
        AIMessage(content="second"),
    ]
    assert last_assistant_text(msgs) == "second"


def test_last_assistant_text_empty():
    assert last_assistant_text([]) == ""
    assert last_assistant_text([HumanMessage(content="only user")]) == ""


def test_last_assistant_text_list_content_blocks():
    msgs = [
        AIMessage(
            content=[
                {"type": "text", "text": "from blocks"},
            ]
        ),
    ]
    assert last_assistant_text(msgs) == "from blocks"


def test_history_lines_empty():
    assert _history_lines([]) == "(no prior messages)"


def test_history_lines_formats_roles():
    text = _history_lines(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
        ]
    )
    assert "user: a" in text
    assert "assistant: b" in text


def test_tools_manifest_includes_registered_tools():
    manifest = _tools_manifest()
    for t in TOOLS:
        assert t.name in manifest


def test_build_executor_messages_order_and_plan_in_system():
    state = {
        "user_prompt": "Do it",
        "attachment_context": "ctx here",
        "chat_history": [
            {"role": "user", "content": "earlier"},
            {"role": "assistant", "content": "ok"},
        ],
        "plan": {"goal_summary": "G", "steps": [], "final_output_description": "F"},
    }
    messages = _build_executor_messages(state)  # type: ignore[arg-type]
    assert isinstance(messages[0], SystemMessage)
    sys_text = messages[0].content
    assert "goal_summary" in sys_text
    assert '"G"' in sys_text or "G" in sys_text
    assert "ctx here" in sys_text
    assert isinstance(messages[1], HumanMessage)
    assert messages[1].content == "earlier"
    assert isinstance(messages[2], AIMessage)
    assert messages[2].content == "ok"
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "Do it"
