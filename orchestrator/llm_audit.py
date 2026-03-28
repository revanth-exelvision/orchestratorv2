from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import AsyncCallbackHandler
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from orchestrator.logging_setup import get_logger

_log = get_logger("llm_audit")

_file_lock = threading.Lock()


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _audit_file_path(log_dir: Path) -> Path:
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    out = log_dir / "llm"
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{day}.jsonl"


def _serialize_messages(messages: list[list[BaseMessage]]) -> list[list[dict[str, Any]]]:
    return [
        [m.model_dump(mode="json") if isinstance(m, BaseMessage) else {"repr": repr(m)} for m in batch]
        for batch in messages
    ]


def _extract_from_llm_result(response: LLMResult) -> tuple[Any, dict[str, Any] | None]:
    token_usage: dict[str, Any] | None = None
    if response.llm_output and isinstance(response.llm_output, dict):
        tu = response.llm_output.get("token_usage")
        if isinstance(tu, dict):
            token_usage = dict(tu)

    outputs: list[Any] = []
    for gen_list in response.generations:
        for gen in gen_list:
            if isinstance(gen, ChatGeneration):
                msg = gen.message
                if isinstance(msg, BaseMessage):
                    outputs.append(msg.model_dump(mode="json"))
                else:
                    outputs.append({"text": getattr(gen, "text", ""), "repr": repr(msg)})
                if token_usage is None and gen.generation_info and isinstance(gen.generation_info, dict):
                    gi = gen.generation_info.get("token_usage")
                    if isinstance(gi, dict):
                        token_usage = dict(gi)
            else:
                text = getattr(gen, "text", None)
                if text is not None:
                    outputs.append(text)
                else:
                    outputs.append(repr(gen))

    if len(outputs) == 1:
        out_val: Any = outputs[0]
    else:
        out_val = outputs
    return out_val, token_usage


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
    with _file_lock:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(line)


class LLMAuditCallbackHandler(AsyncCallbackHandler):
    """Records chat model prompts, outputs, timing, and metadata to daily JSONL files."""

    def __init__(self) -> None:
        super().__init__()
        self._pending: dict[UUID, dict[str, Any]] = {}

    async def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        invocation_params = kwargs.get("invocation_params")
        if isinstance(invocation_params, dict):
            inv_copy = dict(invocation_params)
        else:
            inv_copy = None

        self._pending[run_id] = {
            "ts_start": _utc_now_iso(),
            "t0": time.perf_counter(),
            "serialized": serialized,
            "input_messages": _serialize_messages(messages),
            "invocation_params": inv_copy,
            "tags": list(tags) if tags else None,
            "metadata": dict(metadata) if metadata else None,
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
        }

    async def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        from orchestrator.config import get_settings

        s = get_settings()
        if not s.llm_audit_enabled:
            self._pending.pop(run_id, None)
            return

        pending = self._pending.pop(run_id, None)
        ts_end = _utc_now_iso()
        duration_ms: float | None = None
        if pending and "t0" in pending:
            duration_ms = (time.perf_counter() - pending["t0"]) * 1000.0

        output, token_usage = _extract_from_llm_result(response)

        model_name = ""
        if pending and isinstance(pending.get("invocation_params"), dict):
            model_name = str(pending["invocation_params"].get("model") or "")
        if not model_name and response.llm_output and isinstance(response.llm_output, dict):
            model_name = str(response.llm_output.get("model_name") or "")

        record: dict[str, Any] = {
            "ts_start": pending["ts_start"] if pending else None,
            "ts_end": ts_end,
            "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": list(tags) if tags else (pending.get("tags") if pending else None),
            "metadata": pending.get("metadata") if pending else None,
            "model": model_name,
            "invocation_params": pending.get("invocation_params") if pending else None,
            "serialized_id": (pending.get("serialized") or {}).get("id") if pending else None,
            "input_messages": pending.get("input_messages") if pending else None,
            "output": output,
            "token_usage": token_usage,
            "error": None,
        }

        path = _audit_file_path(Path(s.log_dir))
        try:
            _append_jsonl(path, record)
        except OSError as e:
            _log.exception("LLM audit write failed: %s", e)

    async def on_llm_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        from orchestrator.config import get_settings

        s = get_settings()
        pending = self._pending.pop(run_id, None)
        if not s.llm_audit_enabled:
            return

        ts_end = _utc_now_iso()
        duration_ms: float | None = None
        if pending and "t0" in pending:
            duration_ms = (time.perf_counter() - pending["t0"]) * 1000.0

        record: dict[str, Any] = {
            "ts_start": pending["ts_start"] if pending else None,
            "ts_end": ts_end,
            "duration_ms": round(duration_ms, 3) if duration_ms is not None else None,
            "run_id": str(run_id),
            "parent_run_id": str(parent_run_id) if parent_run_id else None,
            "tags": list(tags) if tags else (pending.get("tags") if pending else None),
            "metadata": pending.get("metadata") if pending else None,
            "model": "",
            "invocation_params": pending.get("invocation_params") if pending else None,
            "serialized_id": (pending.get("serialized") or {}).get("id") if pending else None,
            "input_messages": pending.get("input_messages") if pending else None,
            "output": None,
            "token_usage": None,
            "error": f"{type(error).__name__}: {error}",
        }

        path = _audit_file_path(Path(s.log_dir))
        try:
            _append_jsonl(path, record)
        except OSError as e:
            _log.exception("LLM audit write failed: %s", e)


def get_llm_runnable_config(phase: str) -> dict[str, Any]:
    """RunnableConfig fragment for ``ainvoke``: audit callbacks plus ``metadata.phase`` for filtering."""
    return {
        "callbacks": [LLMAuditCallbackHandler()],
        "metadata": {"phase": phase},
    }
