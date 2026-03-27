# Orchestrator v3

FastAPI service that runs a **LangGraph** pipeline: structured **plan** (Pydantic via `with_structured_output`) then **`create_react_agent`** execution with LangChain `@tool` definitions.

## Setup (use `.venv` only)

```bash
cd /path/to/orchestrator
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Set `OPENAI_API_KEY` (and optionally `OPENAI_MODEL`). You can use a `.env` file in the project root.

## Run

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API

- `GET /health`
- `POST /orchestrate/plan` — same JSON body as `/orchestrate/json`, but returns only the structured **plan** (no executor)
- `POST /orchestrate/execute` — JSON body with an explicit **`plan`** plus `user_prompt` / `chat_history` / optional `context` / `metadata`; runs only the ReAct **executor**
- `POST /orchestrate/json` — JSON body (`user_prompt`, `chat_history`, optional `model`, `context`, `metadata`)
- `POST /orchestrate` — `multipart/form-data`: form field `payload` (JSON string, same shape) + optional file uploads

OpenAPI: `http://localhost:8000/docs`

## Tests

```bash
source .venv/bin/activate
pip install -e '.[dev]'
pytest
```

## Demos

Example clients live under [`demos/`](demos/README.md); that file documents the **command to run** for each script.
