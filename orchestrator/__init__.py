"""AI orchestrator: FastAPI + LangGraph (injectable tools and named flows via ``create_app``)."""

from orchestrator.flow_registry import DEFAULT_FLOWS, get_flow, list_flow_summaries
from orchestrator.graph import GRAPH, build_compiled_graph, generate_plan, last_assistant_text, run_executor
from orchestrator.main import app, create_app
from orchestrator.tools import DEFAULT_TOOLS, TOOLS

__all__ = [
    "DEFAULT_FLOWS",
    "DEFAULT_TOOLS",
    "TOOLS",
    "GRAPH",
    "app",
    "build_compiled_graph",
    "create_app",
    "generate_plan",
    "get_flow",
    "last_assistant_text",
    "list_flow_summaries",
    "run_executor",
]
