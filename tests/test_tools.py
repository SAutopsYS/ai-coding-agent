"""Tests for tool registry and filesystem/search tools."""

from __future__ import annotations

from pathlib import Path

from agent.config_loader import AgentConfig
from agent.tools.base import ToolRegistry
from agent.tools.factory import build_tool_registry
from agent.tools.filesystem import ApplyPatchTool, ReadFileTool, WriteFileTool
from agent.tools.search import GlobFilesTool, SearchContentTool


def test_tool_registry_register_and_run(tmp_path: Path) -> None:
    (tmp_path / "hello.txt").write_text("hi", encoding="utf-8")
    registry = ToolRegistry()
    registry.register(ReadFileTool(tmp_path))
    result = registry.run("read_file", {"path": "hello.txt"})
    assert result.success
    assert result.output == "hi"


def test_write_and_patch(tmp_path: Path) -> None:
    write = WriteFileTool(tmp_path)
    assert write.run({"path": "a.js", "content": "const x = 1;\n"}).success
    patch = ApplyPatchTool(tmp_path)
    result = patch.run(
        {"path": "a.js", "old_string": "const x = 1;", "new_string": "const x = 2;"}
    )
    assert result.success
    assert (tmp_path / "a.js").read_text(encoding="utf-8") == "const x = 2;\n"


def test_glob_and_content_search(tmp_path: Path) -> None:
    (tmp_path / "app").mkdir()
    (tmp_path / "app" / "main.py").write_text("def search():\n    return 1\n", encoding="utf-8")
    (tmp_path / "node_modules" / "x").mkdir(parents=True)
    (tmp_path / "node_modules" / "x" / "index.js").write_text("ignore", encoding="utf-8")

    globber = GlobFilesTool(tmp_path)
    matches = globber.run({"pattern": "**/*.py"}).data["matches"]
    assert matches == ["app/main.py"]

    search = SearchContentTool(tmp_path)
    result = search.run({"pattern": "def search"})
    assert result.success
    assert "app/main.py" in result.output


def test_factory_registers_expected_tools(tmp_path: Path) -> None:
    registry = build_tool_registry(tmp_path, AgentConfig(shell_enabled=False))
    names = {item["name"] for item in registry.list_tools()}
    assert {
        "read_file",
        "write_file",
        "apply_patch",
        "list_directory",
        "glob_files",
        "search_content",
        "execute_shell",
        "git_status",
        "git_diff",
    } <= names
