"""Integration-style tests for the orchestrator with mock LLM."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from agent.config_loader import AgentConfig
from agent.llm.client import LLMClient, LLMClientConfig
from agent.orchestrator import Orchestrator, OrchestratorConfig
from agent.state import WorkspaceState


FIXTURE_REQUEST = (
    "Improve the application so users can better organise and search their notes."
)


def _copy_notes_app(tmp_path: Path) -> Path:
    # tests/ -> ai-coding-agent/ -> workspace root containing node-easy-notes-app-master/
    workspace = Path(__file__).resolve().parents[2]
    source = workspace / "node-easy-notes-app-master"
    if not (source / "server.js").is_file():
        source = Path(r"C:\Users\HP\Downloads\node-easy-notes-app-master\node-easy-notes-app-master")
    dest = tmp_path / "notes-app"
    shutil.copytree(
        source,
        dest,
        ignore=shutil.ignore_patterns("node_modules", ".git", ".venv", "__pycache__"),
    )
    assert (dest / "server.js").is_file()
    assert (dest / "app" / "models" / "note.model.js").is_file()
    return dest


def test_full_pipeline_with_mock_llm(tmp_path: Path) -> None:
    repo = _copy_notes_app(tmp_path)
    config = AgentConfig(
        llm_provider="mock",
        shell_enabled=False,
        max_steps=10,
    )
    state = WorkspaceState(repo_root=repo, request=FIXTURE_REQUEST)
    orch = Orchestrator(
        state,
        agent_config=config,
        llm_client=LLMClient(LLMClientConfig(provider="mock")),
        config=OrchestratorConfig(max_steps=10, dry_run=False),
    )
    result = orch.run()

    assert result.repository_summary is not None
    assert result.execution_plan is not None
    assert result.final_report is not None
    assert result.files_changed, "expected mock provider to modify files"

    model_text = (repo / "app" / "models" / "note.model.js").read_text(encoding="utf-8")
    assert "tags" in model_text

    controller = (repo / "app" / "controllers" / "note.controller.js").read_text(encoding="utf-8")
    assert "exports.create" in controller
    assert "exports.findAll" in controller
    assert "tags" in controller or "search" in controller.lower() or "req.query" in controller

    routes = (repo / "app" / "routes" / "note.routes.js").read_text(encoding="utf-8")
    assert "app.post('/notes'" in routes or 'app.post("/notes"' in routes
    assert "exports.create" not in routes  # routes file references notes.create
    assert "notes.create" in routes

    report = result.to_report_dict()
    assert "files_modified" in report
    json.dumps(report)  # serializable
