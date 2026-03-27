from __future__ import annotations

from app.models import FlowSummary, OrchestratorPlan, PlanStep

# flow_id -> (title, description, plan)
_FLOWS: dict[str, tuple[str, str, OrchestratorPlan]] = {
    "word_stats": (
        "Word statistics",
        "Count words in the user's message using the word_count tool.",
        OrchestratorPlan(
            goal_summary="Report how many words are in the user's text",
            steps=[
                PlanStep(
                    step_id="1",
                    description="Call word_count with the user's prompt or quoted text",
                    tool_name="word_count",
                    inputs="Text from the user message",
                    expected_output="Word count as an integer",
                ),
            ],
            final_output_description="A brief answer stating the word count",
        ),
    ),
    "echo_smoke": (
        "Echo smoke test",
        "Echo back a short snippet via echo_text (useful for wiring checks).",
        OrchestratorPlan(
            goal_summary="Echo user-provided text back",
            steps=[
                PlanStep(
                    step_id="1",
                    description="Use echo_text with the user's text",
                    tool_name="echo_text",
                    inputs="String to echo",
                    expected_output="Same string returned",
                ),
            ],
            final_output_description="Confirmation that the text was echoed",
        ),
    ),
    "agent_bullets": (
        "Bullet formatter agent",
        "Formats the user's text as a Markdown bullet list using bulletize_text.",
        OrchestratorPlan(
            goal_summary="Convert user text into a clean bullet list",
            steps=[
                PlanStep(
                    step_id="1",
                    description="Call bulletize_text on the user's message or quoted passage",
                    tool_name="bulletize_text",
                    inputs="Raw multi-line text from the user",
                    expected_output="Lines prefixed with '- '",
                ),
            ],
            final_output_description="The bullet list plus a one-line note if input was empty",
        ),
    ),
    "agent_metrics": (
        "Text metrics agent",
        "Reports words, characters, and non-empty lines via text_metrics.",
        OrchestratorPlan(
            goal_summary="Summarize quantitative stats about the user's text",
            steps=[
                PlanStep(
                    step_id="1",
                    description="Run text_metrics on the user's prompt or pasted text",
                    tool_name="text_metrics",
                    inputs="Full text to measure",
                    expected_output="words=…, characters=…, non_empty_lines=…",
                ),
            ],
            final_output_description="A short natural-language summary of the metrics string",
        ),
    ),
    "agent_reverse": (
        "Reverse text agent",
        "Returns the user's text reversed character-wise using reverse_text.",
        OrchestratorPlan(
            goal_summary="Show the user's string in reverse order",
            steps=[
                PlanStep(
                    step_id="1",
                    description="Call reverse_text with the user's text",
                    tool_name="reverse_text",
                    inputs="String to reverse",
                    expected_output="Reversed string",
                ),
            ],
            final_output_description="The reversed text and a brief confirmation",
        ),
    ),
}


def list_flow_summaries() -> list[FlowSummary]:
    return [
        FlowSummary(flow_id=fid, title=meta[0], description=meta[1])
        for fid, meta in sorted(_FLOWS.items(), key=lambda x: x[0])
    ]


def get_flow(flow_id: str) -> OrchestratorPlan | None:
    entry = _FLOWS.get(flow_id)
    return entry[2] if entry else None
