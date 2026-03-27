from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, NotRequired

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.agents import create_agent
from langgraph.graph import END, MessagesState, START, StateGraph

from app.llm.factory import get_chat_model
from app.models import OrchestratorPlan
from app.tools import TOOLS


class OrchestratorState(MessagesState):
    user_prompt: str
    attachment_context: str
    chat_history: list[dict[str, str]]
    model_name: NotRequired[str | None]
    plan: NotRequired[dict | None]


def _history_lines(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no prior messages)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def _tools_manifest() -> str:
    lines = []
    for t in TOOLS:
        desc = t.description or "(no description)"
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


def resolve_model_name(model_name: str | None) -> str:
    from app.config import get_settings

    s = get_settings()
    if model_name and str(model_name).strip():
        return str(model_name).strip()
    return s.openai_model


def _resolve_model(state: OrchestratorState) -> str:
    return resolve_model_name(state.get("model_name"))


@lru_cache(maxsize=32)
def _compiled_agent(model: str):
    llm = get_chat_model(model)
    return create_agent(llm, TOOLS)


def _planning_prompt(
    *,
    user_prompt: str,
    attachment_context: str,
    chat_history: list[dict[str, str]],
) -> str:
    return f"""You are a planning component for an AI orchestrator.
Available tools (name and description):
{_tools_manifest()}

Prior chat (compact):
{_history_lines(chat_history)}

Attachment / extra context:
{attachment_context or "(none)"}

User request:
{user_prompt}

Produce a concise plan. Reference tools by name when a step should use a specific tool.
"""


async def generate_plan(
    *,
    user_prompt: str,
    attachment_context: str = "",
    chat_history: list[dict[str, str]] | None = None,
    model_name: str | None = None,
) -> OrchestratorPlan:
    """Structured plan only (no tool execution). Used by the graph plan node and by POST /orchestrate/plan."""
    history = chat_history or []
    model = resolve_model_name(model_name)
    llm = get_chat_model(model)
    structured = llm.with_structured_output(OrchestratorPlan)
    prompt = _planning_prompt(
        user_prompt=user_prompt,
        attachment_context=attachment_context,
        chat_history=history,
    )
    return await structured.ainvoke([HumanMessage(content=prompt)])


async def plan_node(state: OrchestratorState):
    plan = await generate_plan(
        user_prompt=state["user_prompt"],
        attachment_context=state.get("attachment_context") or "",
        chat_history=state["chat_history"],
        model_name=state.get("model_name"),
    )
    return {"plan": plan.model_dump(mode="json")}


def _build_executor_messages(state: OrchestratorState) -> list:
    plan_dict = state.get("plan") or {}
    plan_text = json.dumps(plan_dict, indent=2)
    sys_content = (
        "You are the execution agent. Follow the plan below; use tools when they help.\n\n"
        f"## Plan\n{plan_text}\n\n"
        f"## Attachment / context\n{state['attachment_context'] or '(none)'}"
    )
    out: list = [SystemMessage(content=sys_content)]
    for m in state["chat_history"]:
        role = str(m.get("role", "user")).lower().strip()
        content = str(m.get("content", ""))
        if role == "system":
            out.append(SystemMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        else:
            out.append(HumanMessage(content=content))
    out.append(HumanMessage(content=state["user_prompt"]))
    return out


async def run_executor(
    *,
    plan: dict[str, Any],
    user_prompt: str,
    attachment_context: str = "",
    chat_history: list[dict[str, str]] | None = None,
    model_name: str | None = None,
) -> list:
    """ReAct agent only (no planning). `plan` is the serialized orchestrator plan dict."""
    state: OrchestratorState = {
        "messages": [],
        "user_prompt": user_prompt,
        "attachment_context": attachment_context,
        "chat_history": chat_history or [],
        "model_name": model_name,
        "plan": plan,
    }
    model = _resolve_model(state)
    agent = _compiled_agent(model)
    messages = _build_executor_messages(state)
    out = await agent.ainvoke({"messages": messages})
    return out["messages"]


async def execute_node(state: OrchestratorState):
    messages = await run_executor(
        plan=state.get("plan") or {},
        user_prompt=state["user_prompt"],
        attachment_context=state.get("attachment_context") or "",
        chat_history=state["chat_history"],
        model_name=state.get("model_name"),
    )
    return {"messages": messages}


def build_compiled_graph():
    g = StateGraph(OrchestratorState)
    g.add_node("plan", plan_node)
    g.add_node("execute", execute_node)
    g.add_edge(START, "plan")
    g.add_edge("plan", "execute")
    g.add_edge("execute", END)
    return g.compile()


GRAPH = build_compiled_graph()


def last_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            c = m.content
            if isinstance(c, str) and c.strip():
                return c
            if isinstance(c, list):
                parts = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                text = "".join(parts).strip()
                if text:
                    return text
    return ""
