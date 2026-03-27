#!/usr/bin/env python3
"""Demo: POST /orchestrate with multipart form (JSON payload + text file upload)."""

from __future__ import annotations

import json
import sys
from io import BytesIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import support


def main() -> None:
    support.banner("Demo: POST /orchestrate (multipart + file)")
    print(f"Base URL: {support.base_url()}")
    support.require_api()
    payload = {
        "user_prompt": (
            "Read the uploaded file content from the attachment context and quote its first line."
        ),
        "chat_history": [],
    }
    payload_json = json.dumps(payload)
    file_content = b"Line one of the demo upload.\nLine two for context.\n"
    support.print_json("Payload (form field)", payload)
    print("\n--- uploaded file (demo_notes.txt) ---")
    print(file_content.decode())

    with support.client() as c:
        r = c.post(
            "/orchestrate",
            data={"payload": payload_json},
            files=[("files", ("demo_notes.txt", BytesIO(file_content), "text/plain"))],
        )

    support.print_json(f"Response {r.status_code}", r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text)
    support.raise_for_status_verbose(r)


if __name__ == "__main__":
    main()
