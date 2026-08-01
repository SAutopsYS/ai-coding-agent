"""Tests for path sandboxing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.safety.sandbox import Sandbox


def test_sandbox_resolve_inside(tmp_path: Path) -> None:
    sandbox = Sandbox(repo_root=tmp_path)
    target = tmp_path / "app" / "main.py"
    target.parent.mkdir()
    target.write_text("x", encoding="utf-8")
    resolved = sandbox.resolve("app/main.py")
    assert resolved == target.resolve()


def test_resolve_rejects_escape(tmp_path: Path) -> None:
    sandbox = Sandbox(repo_root=tmp_path)
    with pytest.raises(ValueError):
        sandbox.resolve("../outside.txt")
