# AI Coding Agent

A Python 3.11+ agent that reads an unknown code repository, plans a change from a product request, applies structured edits, checks the result, and writes a short report.

## What it does

1. Explores the repository structure and stack
2. Builds an execution plan from the request
3. Loads the most relevant files
4. Asks an LLM for structured JSON edit instructions
5. Applies those edits through filesystem tools
6. Runs verification checks
7. Prints a final JSON report

The agent is meant to work on many kinds of projects, not only the sample Notes app.

## Setup

```bash
cd ai-coding-agent
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env
```

Edit `.env` with your LLM settings.

### Gemini (default)

Uses the official Google GenAI Python SDK (`google-genai`).

```bash
AGENT_LLM_PROVIDER=gemini
AGENT_LLM_MODEL=gemini-3.6-flash
AGENT_LLM_API_KEY=YOUR_API_KEY
```

Create an API key in [Google AI Studio](https://aistudio.google.com/apikey).

### Other providers

| Variable | Values |
|----------|--------|
| `AGENT_LLM_PROVIDER` | `gemini`, `openai`, `anthropic`, or `mock` |
| `AGENT_LLM_API_KEY` | API key for the chosen provider |
| `AGENT_LLM_MODEL` | Model id, for example `gemini-3.6-flash` |
| `AGENT_LLM_BASE_URL` | Optional custom API base URL |

`mock` does not call a remote API. It builds deterministic edits from the plan and file excerpts. Useful for local tests.

You can also set defaults in `config/default.yaml`. Environment variables override the YAML file.

## Usage

From the `ai-coding-agent` directory:

```bash
python -m agent \
  --repo ../node-easy-notes-app-master \
  --request "Improve the application so users can better organise and search their notes."
```

Force a provider:

```bash
python -m agent \
  --repo ../node-easy-notes-app-master \
  --request "Improve the application so users can better organise and search their notes." \
  --provider gemini
```

Offline mock run:

```bash
python -m agent \
  --repo ../node-easy-notes-app-master \
  --request "Add search and tags for notes" \
  --provider mock
```

### CLI options

| Option | Meaning |
|--------|---------|
| `--repo` | Path to the target repository |
| `--request` | Product request text |
| `--request-file` | Read the request from a file |
| `--config` | Custom YAML config path |
| `--provider` | `openai`, `anthropic`, `gemini`, or `mock` |
| `--dry-run` | Plan and generate edits without writing files |
| `--report-json` | Write the final report to a JSON file |
| `--verbose` | Debug logging |
| `--max-steps` | Cap the number of act-stage steps |

## Project layout

```
src/agent/
  explorer/     Repository scan and summary
  planner/      Heuristic execution plan
  llm/          Provider clients and JSON schemas
  prompts/      Prompt builders
  tools/        Filesystem, search, shell, git tools
  safety/       Path sandbox and limits
  verify/       Post-edit checks
  summary/      Final report
  orchestrator.py
  cli.py
```

## Tests

```bash
pytest
```

## License

MIT
