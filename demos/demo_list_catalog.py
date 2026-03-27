#!/usr/bin/env python3
"""Demo: list registered tools and named flows (GET /orchestrate/tools, GET /orchestrate/flows)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support


def main() -> None:
    support.banner("Demo: list tools and flows")
    print(f"Base URL: {support.base_url()}")
    support.require_api()

    with support.client() as c:
        r_tools = c.get("/orchestrate/tools")
        r_flows = c.get("/orchestrate/flows")

    support.print_json(f"Tools (HTTP {r_tools.status_code})", r_tools.json())
    support.raise_for_status_verbose(r_tools)

    support.print_json(f"Flows (HTTP {r_flows.status_code})", r_flows.json())
    support.raise_for_status_verbose(r_flows)


if __name__ == "__main__":
    main()
