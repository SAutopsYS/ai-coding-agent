"""Git awareness tools (read-only by default; commit disabled)."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any

from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


def _run_git(repo_root: Path, args: list[str], *, timeout: int = 30) -> ToolResult:
    command = ["git", *args]
    logger.info("Running git command: %s", " ".join(command))
    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return ToolResult(success=False, output="", error="git executable not found")
    except subprocess.TimeoutExpired:
        return ToolResult(success=False, output="", error="git command timed out")
    except OSError as exc:
        return ToolResult(success=False, output="", error=str(exc))

    output = ((completed.stdout or "") + (completed.stderr or "")).strip()
    success = completed.returncode == 0
    return ToolResult(
        success=success,
        output=output or f"(exit {completed.returncode})",
        data={"exit_code": completed.returncode, "args": args},
        error=None if success else f"git exit {completed.returncode}",
    )


class GitStatusTool:
    """Run ``git status`` in the target repo."""

    name: str = "git_status"
    description: str = "Show git working tree status for the target repository."

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        del args
        return _run_git(self.repo_root, ["status", "--porcelain=v1", "-b"])


class GitDiffTool:
    """Run ``git diff`` in the target repo."""

    name: str = "git_diff"
    description: str = "Show git diff for changes in the target repository."

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        cmd = ["diff"]
        if bool(args.get("staged")):
            cmd.append("--staged")
        path = args.get("path")
        if path:
            cmd.extend(["--", str(path)])
        return _run_git(self.repo_root, cmd)


class GitLogTool:
    """Run a short ``git log`` for commit message style context."""

    name: str = "git_log"
    description: str = "Show recent commit messages in the target repository."

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root.resolve()

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        max_count = int(args.get("max_count") or 5)
        return _run_git(
            self.repo_root,
            ["log", f"-n{max_count}", "--oneline", "--decorate"],
        )


class GitCommitTool:
    """Commit is intentionally disabled unless explicitly allowed."""

    name: str = "git_commit"
    description: str = "Commit staged changes (disabled by default)."

    def __init__(self, repo_root: Path, *, allow_commit: bool = False) -> None:
        self.repo_root = repo_root.resolve()
        self.allow_commit = allow_commit

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        if not self.allow_commit:
            return ToolResult(
                success=False,
                output="",
                error="git commit is disabled (auto-commit is not allowed).",
            )
        message = str(args.get("message") or "").strip()
        if not message:
            return ToolResult(success=False, output="", error="message is required")
        add = _run_git(self.repo_root, ["add", "-A"])
        if not add.success:
            return add
        return _run_git(self.repo_root, ["commit", "-m", message])
