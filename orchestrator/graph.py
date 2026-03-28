from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any, NotRequired

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, MessagesState, START, StateGraph

from orchestrator.llm.factory import get_chat_model
from orchestrator.llm_audit import get_llm_runnable_config
from orchestrator.models import OrchestratorPlan
from orchestrator.tools import DEFAULT_TOOLS


class OrchestratorState(MessagesState):
    user_prompt: str
    attachment_context: str
    chat_history: list[dict[str, str]]
    model_name: NotRequired[str | None]
    plan: NotRequired[dict | None]
    tools: NotRequired[list[BaseTool]]


def _resolve_tools(state: OrchestratorState) -> list[BaseTool]:
    t = state.get("tools")
    if t:
        return list(t)
    return list(DEFAULT_TOOLS)


def _history_lines(history: list[dict[str, str]]) -> str:
    if not history:
        return "(no prior messages)"
    return "\n".join(f"{m['role']}: {m['content']}" for m in history)


def _tools_manifest(tools: Sequence[BaseTool]) -> str:
    lines = []
    for t in tools:
        desc = t.description or "(no description)"
        lines.append(f"- {t.name}: {desc}")
    return "\n".join(lines)


def resolve_model_name(model_name: str | None) -> str:
    from orchestrator.config import get_settings

    s = get_settings()
    if model_name and str(model_name).strip():
        return str(model_name).strip()
    return s.openai_model


def _resolve_model(state: OrchestratorState) -> str:
    return resolve_model_name(state.get("model_name"))


_agent_cache: dict[tuple[str, tuple[str, ...]], Any] = {}


def _get_agent(model: str, tools: Sequence[BaseTool]):
    sig = tuple(sorted(t.name for t in tools))
    key = (model, sig)
    cached = _agent_cache.get(key)
    if cached is None:
        llm = get_chat_model(model)
        cached = create_agent(llm, list(tools))
        _agent_cache[key] = cached
    return cached


def _planning_prompt(
    *,
    user_prompt: str,
    attachment_context: str,
    chat_history: list[dict[str, str]],
    tools: Sequence[BaseTool],
) -> str:
    return f"""You are a planning component for an AI orchestrator.
Available tools (name and description):
{_tools_manifest(tools)}

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
    tools: Sequence[BaseTool] | None = None,
) -> OrchestratorPlan:
    """Structured plan only (no tool execution). Used by the graph plan node and by POST /orchestrate/plan."""
    history = chat_history or []
    resolved_tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
    model = resolve_model_name(model_name)
    llm = get_chat_model(model)
    structured = llm.with_structured_output(OrchestratorPlan)
    prompt = _planning_prompt(
        user_prompt=user_prompt,
        attachment_context=attachment_context,
        chat_history=history,
        tools=resolved_tools,
    )
    return await structured.ainvoke(
        [HumanMessage(content=prompt)],
        config=get_llm_runnable_config("plan"),
    )


async def plan_node(state: OrchestratorState):
    tools = _resolve_tools(state)
    plan = await generate_plan(
        user_prompt=state["user_prompt"],
        attachment_context=state.get("attachment_context") or "",
        chat_history=state["chat_history"],
        model_name=state.get("model_name"),
        tools=tools,
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
    tools: Sequence[BaseTool] | None = None,
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
    resolved_tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
    model = _resolve_model(state)
    agent = _get_agent(model, resolved_tools)
    messages = _build_executor_messages(state)
    out = await agent.ainvoke(
        {"messages": messages},
        config=get_llm_runnable_config("execute"),
    )
    return out["messages"]


async def execute_node(state: OrchestratorState):
    tools = _resolve_tools(state)
    messages = await run_executor(
        plan=state.get("plan") or {},
        user_prompt=state["user_prompt"],
        attachment_context=state.get("attachment_context") or "",
        chat_history=state["chat_history"],
        model_name=state.get("model_name"),
        tools=tools,
    )
    return {"messages": messages}


def build_compiled_graph() -> Any:
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


def serialize_executor_messages(messages: list[Any]) -> list[dict[str, Any]]:
    """Turn LangChain messages from the executor into JSON-safe dicts for API responses."""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, BaseMessage):
            out.append(m.model_dump(mode="json", exclude_none=True))
        else:
            out.append({"type": "unknown", "repr": repr(m)})
    return out
