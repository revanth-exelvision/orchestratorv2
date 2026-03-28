from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from orchestrator.llm_audit import LLMAuditCallbackHandler, _append_jsonl, _audit_file_path


def test_get_logger_namespaces_under_orchestrator():
    from orchestrator.logging_setup import get_logger

    assert get_logger("foo").name == "orchestrator.foo"
    assert get_logger("orchestrator.bar").name == "orchestrator.bar"


def test_configure_logging_idempotent():
    import orchestrator.logging_setup as ls

    ls.configure_logging()
    ls.configure_logging()


def test_configure_logging_respects_log_level(monkeypatch: pytest.MonkeyPatch):
    import orchestrator.logging_setup as ls

    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    from orchestrator.config import get_settings

    get_settings.cache_clear()
    ls._CONFIGURED = False
    ls.configure_logging()
    root = logging.getLogger("orchestrator")
    assert root.level == logging.DEBUG


@pytest.mark.asyncio
async def test_llm_audit_writes_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_AUDIT_ENABLED", "true")
    from orchestrator.config import get_settings

    get_settings.cache_clear()

    run_id = uuid.uuid4()
    handler = LLMAuditCallbackHandler()
    await handler.on_chat_model_start(
        {"id": ["langchain", "chat_models", "openai", "ChatOpenAI"]},
        [[HumanMessage(content="hello")]],
        run_id=run_id,
        tags=["audit"],
        metadata={"phase": "plan"},
        invocation_params={"model": "gpt-4o-mini", "temperature": 0.0},
    )
    gen = ChatGeneration(
        message=AIMessage(content="done"),
        generation_info={"token_usage": {"total_tokens": 10}},
    )
    result = LLMResult(
        generations=[[gen]],
        llm_output={"token_usage": {"total_tokens": 10}, "model_name": "gpt-4o-mini"},
    )
    await handler.on_llm_end(result, run_id=run_id)

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = tmp_path / "llm" / f"{day}.jsonl"
    assert path.is_file()
    line = path.read_text(encoding="utf-8").strip().splitlines()[-1]
    data = json.loads(line)
    assert data["error"] is None
    assert data["metadata"] == {"phase": "plan"}
    assert data["model"] == "gpt-4o-mini"
    assert data["output"] is not None
    assert data["duration_ms"] is not None


@pytest.mark.asyncio
async def test_llm_audit_skips_when_disabled(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.setenv("LOG_DIR", str(tmp_path))
    monkeypatch.setenv("LLM_AUDIT_ENABLED", "false")
    from orchestrator.config import get_settings

    get_settings.cache_clear()

    run_id = uuid.uuid4()
    handler = LLMAuditCallbackHandler()
    await handler.on_chat_model_start(
        {"id": ["x"]},
        [[HumanMessage(content="a")]],
        run_id=run_id,
    )
    gen = ChatGeneration(message=AIMessage(content="b"))
    result = LLMResult(generations=[[gen]])
    await handler.on_llm_end(result, run_id=run_id)

    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    path = tmp_path / "llm" / f"{day}.jsonl"
    assert not path.exists()


def test_append_jsonl_creates_parent_dirs(tmp_path: Path):
    p = tmp_path / "nested" / "out.jsonl"
    _append_jsonl(p, {"k": 1})
    assert p.read_text(encoding="utf-8").strip() == '{"k": 1}'


def test_audit_file_path_utc_date(tmp_path: Path):
    p = _audit_file_path(tmp_path)
    assert p.parent.name == "llm"
    assert len(p.stem) == 10 and p.stem[4] == "-" and p.suffix == ".jsonl"
