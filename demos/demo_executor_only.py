#!/usr/bin/env python3
"""Demo: POST /orchestrate/execute — ReAct executor with a fixed plan (no planner)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support


def main() -> None:
    support.banner("Demo: POST /orchestrate/execute (executor only)")
    print(f"Base URL: {support.base_url()}")
    support.require_api()
    payload = {
        "plan": {
            "goal_summary": "Count words in a fixed phrase using the word_count tool",
            "steps": [
                {
                    "step_id": "1",
                    "description": "Call word_count on the phrase provided by the user",
                    "tool_name": "word_count",
                    "inputs": "Exact sentence from user_prompt",
                    "expected_output": "Integer word count",
                }
            ],
            "final_output_description": "Report the tool result to the user",
        },
        "user_prompt": (
            'Run word_count on this sentence and tell me the number: '
            '"one two three four five"'
        ),
        "chat_history": [],
        "context": {"demo": "executor_only"},
    }
    support.print_json("Request body", payload)
    with support.client() as c:
        r = c.post("/orchestrate/execute", json=payload)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    support.print_json(f"Response {r.status_code}", body)
    support.raise_for_status_verbose(r)


if __name__ == "__main__":
    main()
