"""Unit tests for RepositoryExplorer."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from agent.explorer import RepositoryExplorer, RepositorySummary


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


@pytest.fixture
def node_repo(tmp_path: Path) -> Path:
    root = tmp_path / "easy-notes"
    _write(
        root / "package.json",
        json.dumps(
            {
                "name": "node-easy-notes-app",
                "main": "server.js",
                "dependencies": {
                    "express": "^4.16.3",
                    "mongoose": "^5.2.8",
                    "body-parser": "^1.18.3",
                },
            },
            indent=2,
        ),
    )
    _write(root / "README.md", "# EasyNotes\n")
    _write(root / "server.js", "const express = require('express');\n")
    _write(root / "app" / "routes" / "note.routes.js", "module.exports = () => {};\n")
    _write(root / "app" / "controllers" / "note.controller.js", "exports.create = () => {};\n")
    _write(root / "config" / "database.config.js", "module.exports = {};\n")
    _write(root / "Dockerfile", "FROM node:18\n")
    # Noise that must be ignored
    _write(root / "node_modules" / "express" / "index.js", "module.exports = {};\n")
    _write(root / "dist" / "bundle.js", "// built\n")
    _write(root / ".git" / "config", "[core]\n")
    return root


@pytest.fixture
def python_repo(tmp_path: Path) -> Path:
    root = tmp_path / "flask-demo"
    _write(
        root / "pyproject.toml",
        '[project]\nname = "flask-demo"\ndependencies = ["flask>=3.0"]\n',
    )
    _write(root / "requirements.txt", "flask>=3.0\n")
    _write(root / "README.md", "# Flask Demo\n")
    _write(root / "app.py", "from flask import Flask\napp = Flask(__name__)\n")
    _write(root / "src" / "utils.py", "def ping():\n    return 'pong'\n")
    _write(root / ".venv" / "lib" / "site.py", "# venv\n")
    _write(root / "__pycache__" / "app.cpython-311.pyc", "binary")
    return root


@pytest.fixture
def go_repo(tmp_path: Path) -> Path:
    root = tmp_path / "go-service"
    _write(root / "go.mod", "module github.com/example/go-service\n\ngo 1.22\n")
    _write(root / "main.go", "package main\nfunc main() {}\n")
    _write(root / "README.md", "# Go service\n")
    return root


def test_explore_node_repo(node_repo: Path) -> None:
    summary = RepositoryExplorer(node_repo).explore()

    assert isinstance(summary, RepositorySummary)
    assert summary.project_name == "node-easy-notes-app"
    assert summary.project_type == "Node.js"
    assert summary.detected_language == "JavaScript"
    assert summary.framework == "Express"
    assert summary.entry_point == "server.js"
    assert "app" in summary.important_directories
    assert "config" in summary.important_directories

    important_paths = {item.path for item in summary.important_files}
    assert "package.json" in important_paths
    assert "README.md" in important_paths
    assert "Dockerfile" in important_paths
    assert "server.js" in important_paths

    assert "Express" in summary.technology_stack
    assert "Mongoose" in summary.technology_stack
    assert "MongoDB" in summary.technology_stack
    assert summary.repository_map is not None
    assert summary.repository_map.file_count >= 5
    assert "node_modules" not in " ".join(summary.repository_map.files)
    assert "dist/" not in " ".join(summary.repository_map.files) and all(
        not f.startswith("dist/") for f in summary.repository_map.files
    )
    assert summary.repository_tree_summary


def test_explore_ignores_noise_directories(node_repo: Path) -> None:
    summary = RepositoryExplorer(node_repo).explore()
    assert summary.repository_map is not None
    joined = "\n".join(summary.repository_map.files + summary.repository_map.directories)
    assert "node_modules" not in joined
    assert ".git" not in joined
    assert "dist" not in joined


def test_explore_python_repo(python_repo: Path) -> None:
    summary = RepositoryExplorer(python_repo).explore()
    assert summary.project_name == "flask-demo"
    assert summary.project_type == "Python"
    assert summary.detected_language == "Python"
    assert summary.framework == "Flask"
    assert summary.entry_point == "app.py"
    assert "src" in summary.important_directories
    paths = {item.path for item in summary.important_files}
    assert "pyproject.toml" in paths
    assert "requirements.txt" in paths
    assert summary.repository_map is not None
    joined = "\n".join(summary.repository_map.files)
    assert ".venv" not in joined
    assert "__pycache__" not in joined


def test_explore_go_repo(go_repo: Path) -> None:
    summary = RepositoryExplorer(go_repo).explore()
    assert summary.project_name == "go-service"
    assert summary.project_type == "Go"
    assert summary.detected_language == "Go"
    assert summary.entry_point == "main.go"


def test_explore_missing_path_raises(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        RepositoryExplorer(missing).explore()


def test_reads_only_important_files(node_repo: Path) -> None:
    summary = RepositoryExplorer(node_repo).explore()
    read_paths = {item.path for item in summary.important_files if item.content is not None}
    # Controller source exists but is not an "important" manifest/readme/entrypoint.
    assert "app/controllers/note.controller.js" not in read_paths
    assert "package.json" in read_paths


def test_important_file_contents_loaded(node_repo: Path) -> None:
    summary = RepositoryExplorer(node_repo).explore()
    package = next(item for item in summary.important_files if item.path == "package.json")
    assert package.content is not None
    assert "express" in package.content


def test_explorer_logs_info(node_repo: Path, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="agent.explorer"):
        RepositoryExplorer(node_repo).explore()
    assert any("Starting repository exploration" in rec.message for rec in caplog.records)
    assert any("Exploration finished" in rec.message for rec in caplog.records)
