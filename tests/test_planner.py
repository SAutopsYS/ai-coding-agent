"""Unit tests for the deterministic ExecutionPlanner."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from agent.explorer.models import (
    ImportantFile,
    RepositoryMap,
    RepositorySummary,
)
from agent.planner import ExecutionPlan, ExecutionPlanner
from agent.planner.heuristics import detect_intents, infer_role, select_relevant_files


def _node_summary() -> RepositorySummary:
    files = [
        "package.json",
        "Readme.md",
        "server.js",
        "app/controllers/note.controller.js",
        "app/models/note.model.js",
        "app/routes/note.routes.js",
        "config/database.config.js",
    ]
    return RepositorySummary(
        project_name="node-easy-notes-app",
        detected_language="JavaScript",
        project_type="Node.js",
        framework="Express",
        entry_point="server.js",
        important_directories=[
            "app",
            "config",
            "app/controllers",
            "app/models",
            "app/routes",
        ],
        important_files=[
            ImportantFile(path="package.json", role="manifest", content='{"main":"server.js"}'),
            ImportantFile(path="Readme.md", role="readme", content="# Notes"),
            ImportantFile(path="server.js", role="entrypoint", content="listen(3000)"),
        ],
        technology_stack=["Node.js", "JavaScript", "Express", "Mongoose", "MongoDB"],
        repository_tree_summary="node-easy-notes-app/\n├── app/\n└── server.js",
        repository_map=RepositoryMap(
            root="/tmp/node-easy-notes-app",
            files=files,
            directories=["app", "app/controllers", "app/models", "app/routes", "config"],
            file_count=len(files),
            directory_count=5,
        ),
    )


def _python_summary() -> RepositorySummary:
    files = [
        "pyproject.toml",
        "README.md",
        "app.py",
        "src/models/item.py",
        "src/routes/item_routes.py",
        "src/controllers/item_controller.py",
        "src/middleware/auth.py",
    ]
    return RepositorySummary(
        project_name="flask-demo",
        detected_language="Python",
        project_type="Python",
        framework="Flask",
        entry_point="app.py",
        important_directories=["src", "src/models", "src/routes", "src/controllers", "src/middleware"],
        important_files=[
            ImportantFile(path="pyproject.toml", role="manifest", content='name="flask-demo"'),
            ImportantFile(path="README.md", role="readme", content="# Flask"),
            ImportantFile(path="app.py", role="entrypoint", content="app.run()"),
        ],
        technology_stack=["Python", "Flask"],
        repository_map=RepositoryMap(
            root="/tmp/flask-demo",
            files=files,
            directories=["src", "src/models", "src/routes", "src/controllers", "src/middleware"],
            file_count=len(files),
            directory_count=5,
        ),
    )


def test_detect_intents_search_and_organize() -> None:
    intents = detect_intents(
        "Improve the application so users can better organise and search their notes."
    )
    assert "search" in intents
    assert "organize" in intents
    assert "persistence" in intents


def test_detect_intents_authentication() -> None:
    intents = detect_intents("Add JWT authentication middleware for protected routes")
    assert "authentication" in intents


def test_infer_role_patterns() -> None:
    assert infer_role("app/models/note.model.js") == "model"
    assert infer_role("app/controllers/note.controller.js") == "controller"
    assert infer_role("app/routes/note.routes.js") == "route"
    assert infer_role("src/middleware/auth.py") == "middleware"


def test_search_prioritizes_controllers_and_routes() -> None:
    summary = _node_summary()
    intents = detect_intents("Add search for notes by title")
    relevant = select_relevant_files(summary, "Add search for notes by title", intents)
    by_path = {item.path: item for item in relevant}

    assert "app/controllers/note.controller.js" in by_path
    assert "app/routes/note.routes.js" in by_path
    assert by_path["app/controllers/note.controller.js"].priority >= by_path[
        "config/database.config.js"
    ].priority
    assert "controller" in by_path["app/controllers/note.controller.js"].reason.lower()


def test_tags_prioritize_models_and_controllers() -> None:
    summary = _node_summary()
    request = "Add tags to notes for organization"
    intents = detect_intents(request)
    relevant = select_relevant_files(summary, request, intents)
    by_path = {item.path: item for item in relevant}

    assert "app/models/note.model.js" in by_path
    assert "app/controllers/note.controller.js" in by_path
    assert by_path["app/models/note.model.js"].priority >= by_path[
        "app/controllers/note.controller.js"
    ].priority
    assert "model" in by_path["app/models/note.model.js"].reason.lower()


def test_create_plan_structure_for_organise_and_search() -> None:
    summary = _node_summary()
    request = (
        "Improve the application so users can better organise and search their notes."
    )
    plan = ExecutionPlanner().create_plan(summary, request)

    assert isinstance(plan, ExecutionPlan)
    assert plan.goal
    assert plan.request == request
    assert plan.assumptions
    assert plan.relevant_files
    assert plan.steps
    assert plan.risks
    assert plan.validation_checklist
    assert plan.unchanged_functionality
    assert "search" in plan.detected_intents
    assert "organize" in plan.detected_intents

    # Every relevant file must explain why.
    for item in plan.relevant_files:
        assert item.reason.strip()
        assert item.path

    step_titles = " ".join(step.title.lower() for step in plan.steps)
    assert "model" in step_titles or "organization" in step_titles
    assert "search" in step_titles
    assert "route" in step_titles

    assert any(item.category == "regression" for item in plan.validation_checklist)
    assert any(
        "create/read/update/delete" in item.description.lower()
        for item in plan.validation_checklist
    )
    assert any("existing http methods" in text.lower() for text in plan.unchanged_functionality)
    assert any("list/get-by-id" in text.lower() for text in plan.unchanged_functionality)


def test_auth_plan_includes_middleware_for_python_repo() -> None:
    summary = _python_summary()
    plan = ExecutionPlanner().create_plan(
        summary,
        "Add authentication middleware to protect item routes",
    )
    paths = {item.path for item in plan.relevant_files}
    assert "src/middleware/auth.py" in paths
    assert "authentication" in plan.detected_intents
    assert any("middleware" in step.title.lower() or "auth" in step.title.lower() for step in plan.steps)


def test_empty_request_raises() -> None:
    with pytest.raises(ValueError):
        ExecutionPlanner().create_plan(_node_summary(), "   ")


def test_planner_does_not_mutate_summary_files(tmp_path: Path) -> None:
    """Planner must not write to disk."""
    summary = _node_summary()
    before = list(tmp_path.iterdir())
    ExecutionPlanner().create_plan(
        summary,
        "Add search and tags for notes",
    )
    after = list(tmp_path.iterdir())
    assert before == after


def test_planner_logs_info(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.INFO, logger="agent.planner"):
        ExecutionPlanner().create_plan(
            _node_summary(),
            "Add search for notes",
        )
    assert any("Creating execution plan" in rec.message for rec in caplog.records)
    assert any("Plan created" in rec.message for rec in caplog.records)
