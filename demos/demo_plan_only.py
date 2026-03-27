#!/usr/bin/env python3
"""Demo: POST /orchestrate/plan — structured plan only (no ReAct / tool execution)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support


def main() -> None:
    support.banner("Demo: POST /orchestrate/plan (plan generation only)")
    print(f"Base URL: {support.base_url()}")
    support.require_api()
    payload = {
        "user_prompt": (
            "I need to analyze a short phrase: break it into steps, and say when "
            "word_count or echo_text would be useful."
        ),
        "chat_history": [
            {"role": "user", "content": "Keep the plan brief — at most 3 steps."},
        ],
        "context": {"demo": "plan_only"},
    }
    support.print_json("Request body", payload)
    with support.client() as c:
        r = c.post("/orchestrate/plan", json=payload)
    body = r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    support.print_json(f"Response {r.status_code}", body)
    support.raise_for_status_verbose(r)


if __name__ == "__main__":
    main()
