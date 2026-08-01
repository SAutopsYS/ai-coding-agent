"""Unit tests for IgnoreMatcher."""

from __future__ import annotations

from pathlib import Path

from agent.utils.ignore import IgnoreMatcher


def test_ignores_default_directory_names(tmp_path: Path) -> None:
    matcher = IgnoreMatcher.from_default_patterns()
    repo = tmp_path / "repo"
    repo.mkdir()

    ignored = [
        repo / ".git" / "config",
        repo / "node_modules" / "pkg" / "index.js",
        repo / "dist" / "bundle.js",
        repo / "build" / "out.bin",
        repo / "pkg" / "__pycache__" / "x.pyc",
        repo / ".venv" / "lib" / "site.py",
    ]
    for path in ignored:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
        assert matcher.is_ignored(path, repo_root=repo) is True


def test_does_not_ignore_normal_source_files(tmp_path: Path) -> None:
    matcher = IgnoreMatcher.from_default_patterns()
    repo = tmp_path / "repo"
    source = repo / "app" / "main.py"
    source.parent.mkdir(parents=True)
    source.write_text("print('hi')", encoding="utf-8")
    assert matcher.is_ignored(source, repo_root=repo) is False


def test_path_outside_repo_is_ignored(tmp_path: Path) -> None:
    matcher = IgnoreMatcher.from_default_patterns()
    repo = tmp_path / "repo"
    repo.mkdir()
    outside = tmp_path / "other" / "file.txt"
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")
    assert matcher.is_ignored(outside, repo_root=repo) is True
