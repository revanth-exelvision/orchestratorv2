#!/usr/bin/env python3
"""Demo: GET /health (no API key required)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import httpx

import support


def main() -> None:
    support.banner("Demo: GET /health")
    print(f"Base URL: {support.base_url()}")
    try:
        with support.client() as c:
            r = c.get("/health")
    except httpx.ConnectError:
        raise SystemExit(support.connect_help()) from None
    support.print_json(f"status {r.status_code}", r.json())


if __name__ == "__main__":
    main()
