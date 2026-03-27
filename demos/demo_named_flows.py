#!/usr/bin/env python3
"""Demo: GET /orchestrate/flows and POST /orchestrate/flows/{flow_id} (sample agents; needs OPENAI_API_KEY)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support


def main() -> None:
    support.banner("Demo: named flows (sample agents)")
    print(f"Base URL: {support.base_url()}")
    support.require_api()

    with support.client() as c:
        listed = c.get("/orchestrate/flows")
    support.print_json(f"GET /orchestrate/flows {listed.status_code}", listed.json())
    support.raise_for_status_verbose(listed)

    flow_id = "agent_metrics"
    body = {
        "user_prompt": "Measure this text:\n\nhello world\nsecond line",
        "chat_history": [],
    }
    support.print_json(f"POST /orchestrate/flows/{flow_id} body", body)
    with support.client() as c:
        r = c.post(f"/orchestrate/flows/{flow_id}", json=body)
    support.print_json(
        f"Response {r.status_code}",
        r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text,
    )
    support.raise_for_status_verbose(r)


if __name__ == "__main__":
    main()
