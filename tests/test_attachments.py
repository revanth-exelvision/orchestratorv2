from __future__ import annotations

from io import BytesIO

import pytest
from starlette.datastructures import UploadFile

from orchestrator.attachments import format_context_block, normalize_uploads
from orchestrator.config import Settings


@pytest.mark.asyncio
async def test_normalize_uploads_text_file():
    settings = Settings(
        max_upload_files=5,
        max_upload_bytes_per_file=10_000,
        max_total_request_bytes=50_000,
    )
    uf = UploadFile(
        filename="note.txt",
        file=BytesIO(b"hello world"),
        headers={"content-type": "text/plain"},
    )
    text, total = await normalize_uploads([uf], settings)
    assert "note.txt" in text
    assert "hello world" in text
    assert total == 11


@pytest.mark.asyncio
async def test_normalize_uploads_binary_skips_content():
    settings = Settings(
        max_upload_files=5,
        max_upload_bytes_per_file=10_000,
        max_total_request_bytes=50_000,
    )
    uf = UploadFile(
        filename="blob.bin",
        file=BytesIO(b"\x00\xff\x00"),
        headers={"content-type": "application/octet-stream"},
    )
    text, total = await normalize_uploads([uf], settings)
    assert "blob.bin" in text
    assert "Binary or non-text" in text
    assert "3 bytes" in text
    assert total == 3


@pytest.mark.asyncio
async def test_normalize_uploads_too_many_files():
    settings = Settings(max_upload_files=2, max_upload_bytes_per_file=1000, max_total_request_bytes=10_000)
    files = [
        UploadFile(filename=f"{i}.txt", file=BytesIO(b"x"), headers={"content-type": "text/plain"})
        for i in range(3)
    ]
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await normalize_uploads(files, settings)
    assert exc.value.status_code == 400
    assert "Too many files" in exc.value.detail


@pytest.mark.asyncio
async def test_normalize_uploads_file_too_large():
    settings = Settings(max_upload_files=5, max_upload_bytes_per_file=5, max_total_request_bytes=100)
    uf = UploadFile(
        filename="big.txt",
        file=BytesIO(b"123456"),
        headers={"content-type": "text/plain"},
    )
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as exc:
        await normalize_uploads([uf], settings)
    assert exc.value.status_code == 400


def test_format_context_block_empty():
    assert format_context_block("", None, None) == ""


def test_format_context_block_with_context_and_files():
    block = format_context_block(
        "### File: a.txt (text/plain)\nhi",
        {"foo": 1},
        {"trace": "x"},
    )
    assert "Client context" in block
    assert "foo" in block
    assert "Client metadata" in block
    assert "trace" in block
    assert "Uploaded files" in block
    assert "a.txt" in block
