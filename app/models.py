from typing import Any

from pydantic import BaseModel, Field


class ChatMessageItem(BaseModel):
    role: str = Field(..., description="user | assistant | system")
    content: str


class OrchestratePayload(BaseModel):
    user_prompt: str
    chat_history: list[ChatMessageItem] = Field(default_factory=list)
    model: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class PlanStep(BaseModel):
    step_id: str
    description: str
    tool_name: str | None = Field(None, description="Tool to use for this step, if any")
    inputs: str = Field("", description="Inputs this step needs")
    expected_output: str = Field("", description="What this step should produce")


class OrchestratorPlan(BaseModel):
    goal_summary: str
    steps: list[PlanStep] = Field(default_factory=list)
    final_output_description: str = Field(
        "",
        description="What the overall request should deliver to the user",
    )


class OrchestrateResponse(BaseModel):
    plan: OrchestratorPlan
    answer: str = Field("", description="Final assistant text from the executor agent")


class FlowSummary(BaseModel):
    flow_id: str
    title: str = ""
    description: str = ""


class NamedFlowExecutePayload(BaseModel):
    """Body for named pre-defined plan execution (plan is resolved server-side by flow_id)."""

    user_prompt: str
    chat_history: list[ChatMessageItem] = Field(default_factory=list)
    model: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ExecutePayload(BaseModel):
    """Body for executor-only: supply a plan (e.g. from /orchestrate/plan) plus the user turn and history."""

    plan: OrchestratorPlan
    user_prompt: str
    chat_history: list[ChatMessageItem] = Field(default_factory=list)
    model: str | None = None
    context: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None


class ExecuteResponse(BaseModel):
    answer: str = Field("", description="Final assistant text from the ReAct executor")
