from __future__ import annotations

import json
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from pydantic import ValidationError

from app.attachments import format_context_block, normalize_uploads
from app.config import Settings, get_settings
from app.flow_registry import get_flow, list_flow_summaries
from app.graph import GRAPH, generate_plan, last_assistant_text, run_executor
from app.models import (
    ExecutePayload,
    ExecuteResponse,
    FlowSummary,
    NamedFlowExecutePayload,
    OrchestratePayload,
    OrchestrateResponse,
    OrchestratorPlan,
)

app = FastAPI(title="Orchestrator", version="0.1.0")


@app.get("/health")
def health():
    return {"status": "ok"}


def _payload_from_json_str(raw: str) -> OrchestratePayload:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, detail=f"Invalid JSON payload: {e}") from e
    try:
        return OrchestratePayload.model_validate(data)
    except ValidationError as e:
        raise HTTPException(400, detail=e.errors()) from e


async def _attachment_context_from_parts(
    files: list[UploadFile],
    settings: Settings,
    context: dict | None,
    metadata: dict | None,
) -> str:
    upload_block, _ = await normalize_uploads(files, settings)
    return format_context_block(upload_block, context, metadata)


async def _attachment_context_for_payload(
    body: OrchestratePayload,
    files: list[UploadFile],
    settings: Settings,
) -> str:
    return await _attachment_context_from_parts(
        files, settings, body.context, body.metadata
    )


async def _run_orchestration(body: OrchestratePayload, files: list[UploadFile], settings: Settings):
    attachment_context = await _attachment_context_for_payload(body, files, settings)
    state = {
        "messages": [],
        "user_prompt": body.user_prompt,
        "attachment_context": attachment_context,
        "chat_history": [m.model_dump() for m in body.chat_history],
        "model_name": body.model,
    }
    out = await GRAPH.ainvoke(state)
    plan = OrchestratorPlan.model_validate(out["plan"])
    answer = last_assistant_text(out.get("messages", []))
    return OrchestrateResponse(plan=plan, answer=answer)


@app.post("/orchestrate/execute", response_model=ExecuteResponse)
async def orchestrate_execute_only(
    body: ExecutePayload,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Run only the ReAct executor with a supplied plan (no planner call)."""
    attachment_context = await _attachment_context_from_parts(
        [], settings, body.context, body.metadata
    )
    messages = await run_executor(
        plan=body.plan.model_dump(mode="json"),
        user_prompt=body.user_prompt,
        attachment_context=attachment_context,
        chat_history=[m.model_dump() for m in body.chat_history],
        model_name=body.model,
    )
    return ExecuteResponse(answer=last_assistant_text(messages))


@app.get("/orchestrate/flows", response_model=list[FlowSummary])
def list_orchestrate_flows():
    """List ids and metadata for server-defined plans invokable via POST /orchestrate/flows/{flow_id}."""
    return list_flow_summaries()


@app.post("/orchestrate/flows/{flow_id}", response_model=ExecuteResponse)
async def orchestrate_named_flow(
    flow_id: str,
    body: NamedFlowExecutePayload,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Run the ReAct executor with a pre-configured plan looked up by `flow_id`."""
    plan = get_flow(flow_id)
    if plan is None:
        raise HTTPException(404, detail=f"Unknown flow_id: {flow_id}")
    attachment_context = await _attachment_context_from_parts(
        [], settings, body.context, body.metadata
    )
    messages = await run_executor(
        plan=plan.model_dump(mode="json"),
        user_prompt=body.user_prompt,
        attachment_context=attachment_context,
        chat_history=[m.model_dump() for m in body.chat_history],
        model_name=body.model,
    )
    return ExecuteResponse(answer=last_assistant_text(messages))


@app.post("/orchestrate/plan", response_model=OrchestratorPlan)
async def orchestrate_plan_only(
    body: OrchestratePayload,
    settings: Annotated[Settings, Depends(get_settings)],
):
    """Run only the planning LLM (structured `OrchestratorPlan`); does not invoke the ReAct executor."""
    attachment_context = await _attachment_context_for_payload(body, [], settings)
    return await generate_plan(
        user_prompt=body.user_prompt,
        attachment_context=attachment_context,
        chat_history=[m.model_dump() for m in body.chat_history],
        model_name=body.model,
    )


@app.post("/orchestrate/json", response_model=OrchestrateResponse)
async def orchestrate_json(
    body: OrchestratePayload,
    settings: Annotated[Settings, Depends(get_settings)],
):
    return await _run_orchestration(body, [], settings)


@app.post("/orchestrate", response_model=OrchestrateResponse)
async def orchestrate_multipart(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    payload: str = Form(..., description="JSON string matching OrchestratePayload"),
    files: list[UploadFile] | None = File(None),
):
    cl = request.headers.get("content-length")
    if cl and cl.isdigit() and int(cl) > settings.max_total_request_bytes:
        raise HTTPException(413, detail="Request body too large")

    body = _payload_from_json_str(payload)
    file_list = files or []
    return await _run_orchestration(body, file_list, settings)


def run():
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
