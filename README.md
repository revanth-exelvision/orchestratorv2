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
uvicorn orchestrator.main:app --reload --host 0.0.0.0 --port 8000
```

## API

- `GET /health`
- `GET /orchestrate/tools` — registered tool names and descriptions (matches `create_app(tools=...)`)
- `GET /orchestrate/flows` — named flow ids and metadata (matches `create_app(flows=...)`)
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

## Build and use in another project

### Build distributable artifacts

From this repo root, install the standard build tool once (any environment), then produce a wheel and source distribution:

```bash
pip install build
python -m build
```

Outputs land in `dist/`, for example `orchestrator-0.1.0-py3-none-any.whl` and `orchestrator-0.1.0.tar.gz`. Bump `version` in `pyproject.toml` when you cut a new release.

### Install into another project

**From the wheel (typical for a local or internal handoff):**

```bash
pip install /absolute/path/to/orchestratorv3/dist/orchestrator-0.1.0-py3-none-any.whl
```

**From the source tree (editable, good while both projects change):**

```bash
pip install -e /absolute/path/to/orchestratorv3
```

**From a Git URL (after you push the repo):**

```bash
pip install "git+https://example.com/your-org/orchestratorv3.git@v0.1.0"
```

After installation, import the package as `orchestrator` (see `orchestrator/` in this repo). The console entry point `orchestrator-serve` is also available if you need the bundled server runner.

### Declare it as a dependency (consumer `pyproject.toml`)

Point at a local path during development:

```toml
[project]
dependencies = [
    "orchestrator @ file:///absolute/path/to/orchestratorv3",
]
```

Or at a built wheel:

```toml
dependencies = [
    "orchestrator @ file:///absolute/path/to/orchestratorv3/dist/orchestrator-0.1.0-py3-none-any.whl",
]
```

For a published package, use a normal version constraint once the project is on an index you configure (for example PyPI or a private package index).

## Demos

Example clients live under [`demos/`](demos/README.md); that file documents the **command to run** for each script.
