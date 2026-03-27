"""Shared helpers for demo scripts (run from repo root: python demos/<script>.py)."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

try:
    import httpx
except ImportError as e:
    raise SystemExit(
        "Demos require httpx. Install with: pip install httpx\n"
        "Or: pip install -e '.[dev]'"
    ) from e

DEFAULT_BASE = os.environ.get("ORCHESTRATOR_URL", "http://127.0.0.1:8000").rstrip("/")
DEFAULT_TIMEOUT = float(os.environ.get("ORCHESTRATOR_DEMO_TIMEOUT", "120"))


def connect_help() -> str:
    return (
        f"Cannot connect to {DEFAULT_BASE}.\n\n"
        "Start the API from the repository root:\n"
        "  source .venv/bin/activate\n"
        "  export OPENAI_API_KEY=sk-...   # required for /orchestrate/plan and related routes\n"
        "  uvicorn orchestrator.main:app --host 127.0.0.1 --port 8000\n"
    )


def require_api() -> None:
    """Exit with instructions if nothing is listening (avoids a long httpx timeout + traceback)."""
    try:
        with httpx.Client(
            base_url=DEFAULT_BASE,
            timeout=httpx.Timeout(5.0, connect=2.0),
        ) as c:
            c.get("/health")
    except httpx.ConnectError:
        raise SystemExit(connect_help()) from None


def base_url() -> str:
    return DEFAULT_BASE


def client() -> httpx.Client:
    return httpx.Client(base_url=DEFAULT_BASE, timeout=DEFAULT_TIMEOUT)


def raise_for_status_verbose(response: httpx.Response) -> None:
    if response.is_success:
        return
    print(f"\nHTTP {response.status_code} — request failed.", file=sys.stderr)
    try:
        detail = response.json()
        print(json.dumps(detail, indent=2, default=str), file=sys.stderr)
    except Exception:
        print(response.text[:4000], file=sys.stderr)
    if response.status_code in (401, 403) or "api_key" in response.text.lower():
        print(
            "\nHint: the server needs a valid OPENAI_API_KEY in its environment.",
            file=sys.stderr,
        )
    response.raise_for_status()


def banner(title: str) -> None:
    line = "=" * 64
    print(f"\n{line}\n{title}\n{line}")


def print_json(label: str, data: Any) -> None:
    print(f"\n--- {label} ---")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=2, default=str))
    else:
        print(data)
