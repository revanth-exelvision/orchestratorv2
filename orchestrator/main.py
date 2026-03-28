from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from langchain_core.tools import BaseTool
from pydantic import ValidationError
from starlette.responses import JSONResponse

from orchestrator.attachments import format_context_block, normalize_uploads
from orchestrator.config import Settings, get_settings
from orchestrator.flow_registry import DEFAULT_FLOWS, get_flow, list_flow_summaries
from orchestrator.graph import (
    GRAPH,
    generate_plan,
    last_assistant_text,
    run_executor,
    serialize_executor_messages,
)
from orchestrator.models import (
    ExecutePayload,
    ExecuteResponse,
    FlowSummary,
    NamedFlowExecutePayload,
    OrchestratePayload,
    OrchestrateResponse,
    OrchestratorPlan,
    ToolSummary,
)
from orchestrator.logging_setup import configure_logging, get_logger
from orchestrator.tools import DEFAULT_TOOLS

logger = get_logger(__name__)


def create_app(
    *,
    tools: Sequence[BaseTool] | None = None,
    flows: Mapping[str, tuple[str, str, OrchestratorPlan]] | None = None,
) -> FastAPI:
    """Build a FastAPI app with configurable LangChain tools and named-flow registry.

    Host projects can ``pip install`` this package and call ``create_app(tools=my_tools, flows=my_flows)``
    to serve orchestration with domain-specific tools and plans.
    """
    configure_logging()
    app = FastAPI(title="Orchestrator", version="0.1.0")
    app.state.tools = list(tools) if tools is not None else list(DEFAULT_TOOLS)
    app.state.flow_registry = dict(flows) if flows is not None else dict(DEFAULT_FLOWS)

    @app.middleware("http")
    async def log_unhandled_exceptions(request: Request, call_next):
        try:
            return await call_next(request)
        except HTTPException:
            raise
        except RequestValidationError:
            raise
        except Exception:
            logger.exception("Unhandled error on %s %s", request.method, request.url.path)
            return JSONResponse(
                status_code=500,
                content={"detail": "Internal server error"},
            )

    @app.get("/health")
    def health():
        return {"status": "ok"}

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

    async def _run_orchestration(
        body: OrchestratePayload,
        files: list[UploadFile],
        settings: Settings,
        request: Request,
    ):
        attachment_context = await _attachment_context_for_payload(body, files, settings)
        state = {
            "messages": [],
            "user_prompt": body.user_prompt,
            "attachment_context": attachment_context,
            "chat_history": [m.model_dump() for m in body.chat_history],
            "model_name": body.model,
            "tools": request.app.state.tools,
        }
        out = await GRAPH.ainvoke(state)
        plan = OrchestratorPlan.model_validate(out["plan"])
        exec_messages = out.get("messages", [])
        answer = last_assistant_text(exec_messages)
        return OrchestrateResponse(
            plan=plan,
            answer=answer,
            messages=serialize_executor_messages(exec_messages),
        )

    def _payload_from_json_str(raw: str) -> OrchestratePayload:
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise HTTPException(400, detail=f"Invalid JSON payload: {e}") from e
        try:
            return OrchestratePayload.model_validate(data)
        except ValidationError as e:
            raise HTTPException(400, detail=e.errors()) from e

    @app.post("/orchestrate/execute", response_model=ExecuteResponse)
    async def orchestrate_execute_only(
        body: ExecutePayload,
        request: Request,
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
            tools=request.app.state.tools,
        )
        return ExecuteResponse(
            answer=last_assistant_text(messages),
            messages=serialize_executor_messages(messages),
        )

    @app.get("/orchestrate/tools", response_model=list[ToolSummary])
    def list_orchestrate_tools(request: Request):
        """List tools registered on this app (same set used by plan and execute)."""
        return [
            ToolSummary(name=t.name, description=(t.description or "").strip())
            for t in request.app.state.tools
        ]

    @app.get("/orchestrate/flows", response_model=list[FlowSummary])
    def list_orchestrate_flows(request: Request):
        """List ids and metadata for server-defined plans invokable via POST /orchestrate/flows/{flow_id}."""
        return list_flow_summaries(request.app.state.flow_registry)

    @app.post("/orchestrate/flows/{flow_id}", response_model=ExecuteResponse)
    async def orchestrate_named_flow(
        flow_id: str,
        body: NamedFlowExecutePayload,
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
    ):
        """Run the ReAct executor with a pre-configured plan looked up by `flow_id`."""
        plan = get_flow(flow_id, request.app.state.flow_registry)
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
            tools=request.app.state.tools,
        )
        return ExecuteResponse(
            answer=last_assistant_text(messages),
            messages=serialize_executor_messages(messages),
        )

    @app.post("/orchestrate/plan", response_model=OrchestratorPlan)
    async def orchestrate_plan_only(
        body: OrchestratePayload,
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
    ):
        """Run only the planning LLM (structured `OrchestratorPlan`); does not invoke the ReAct executor."""
        attachment_context = await _attachment_context_for_payload(body, [], settings)
        return await generate_plan(
            user_prompt=body.user_prompt,
            attachment_context=attachment_context,
            chat_history=[m.model_dump() for m in body.chat_history],
            model_name=body.model,
            tools=request.app.state.tools,
        )

    @app.post("/orchestrate/json", response_model=OrchestrateResponse)
    async def orchestrate_json(
        body: OrchestratePayload,
        request: Request,
        settings: Annotated[Settings, Depends(get_settings)],
    ):
        return await _run_orchestration(body, [], settings, request)

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
        return await _run_orchestration(body, file_list, settings, request)

    return app


app = create_app()


def run():
    import uvicorn

    uvicorn.run("orchestrator.main:app", host="0.0.0.0", port=8000, reload=False)
