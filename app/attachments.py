import json
from typing import Any

from fastapi import HTTPException, UploadFile

from app.config import Settings


def _is_textual_content_type(content_type: str) -> bool:
    ct = content_type.split(";")[0].strip().lower()
    return ct.startswith("text/") or ct in ("application/json", "application/xml")


async def normalize_uploads(
    files: list[UploadFile],
    settings: Settings,
) -> tuple[str, int]:
    """Read uploads into a single context string for the LLM. Returns (text, total_bytes)."""
    if len(files) > settings.max_upload_files:
        raise HTTPException(
            400,
            detail=f"Too many files (max {settings.max_upload_files})",
        )

    parts: list[str] = []
    total = 0

    for uf in files:
        raw = await uf.read()
        total += len(raw)
        if total > settings.max_total_request_bytes:
            raise HTTPException(400, detail="Total upload size exceeds limit")

        if len(raw) > settings.max_upload_bytes_per_file:
            raise HTTPException(
                400,
                detail=f"File {uf.filename!r} exceeds per-file size limit",
            )

        ct = (uf.content_type or "").split(";")[0].strip().lower()
        name = uf.filename or "unnamed"

        if _is_textual_content_type(ct):
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                text = raw.decode("utf-8", errors="replace")
            parts.append(f"### File: {name} ({ct})\n{text}")
        else:
            parts.append(
                f"### File: {name} ({ct or 'unknown'})\n"
                f"[Binary or non-text; {len(raw)} bytes — not inlined.]"
            )

    return "\n\n".join(parts), total


def format_context_block(
    attachment_block: str,
    context: dict[str, Any] | None,
    metadata: dict[str, Any] | None,
) -> str:
    chunks: list[str] = []
    if context is not None:
        chunks.append("## Client context (JSON)\n```json\n" + json.dumps(context, indent=2) + "\n```")
    if metadata is not None:
        chunks.append("## Client metadata (JSON)\n```json\n" + json.dumps(metadata, indent=2) + "\n```")
    if attachment_block.strip():
        chunks.append("## Uploaded files\n" + attachment_block)
    return "\n\n".join(chunks) if chunks else ""
