from orchestrator.tools.sample_agents import SAMPLE_AGENT_TOOLS
from orchestrator.tools.stubs import TOOLS as _STUB_TOOLS

DEFAULT_TOOLS = [*_STUB_TOOLS, *SAMPLE_AGENT_TOOLS]
TOOLS = DEFAULT_TOOLS

__all__ = ["DEFAULT_TOOLS", "TOOLS"]
