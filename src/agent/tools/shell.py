"""Bounded shell command execution with allowlist."""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path
from typing import Any

from agent.tools.base import ToolResult

logger = logging.getLogger(__name__)


class ShellTool:
    """Execute an allowlisted shell command in the target repository."""

    name: str = "execute_shell"
    description: str = "Run an allowlisted shell command in the repository root."

    def __init__(
        self,
        repo_root: Path,
        *,
        allowlist: list[str] | None = None,
        timeout_seconds: int = 60,
        enabled: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.allowlist = [item.strip() for item in (allowlist or []) if item.strip()]
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        if not self.enabled:
            return ToolResult(
                success=False,
                output="",
                error="Shell execution is disabled in configuration.",
            )

        command = str(args.get("command") or "").strip()
        if not command:
            return ToolResult(success=False, output="", error="command is required")

        if not self._is_allowed(command):
            return ToolResult(
                success=False,
                output="",
                error=f"Command not allowlisted: {command}",
                data={"allowlist": self.allowlist},
            )

        logger.info("Executing shell command: %s", command)
        try:
            completed = subprocess.run(
                command,
                shell=True,
                cwd=str(self.repo_root),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output="",
                error=f"Command timed out after {self.timeout_seconds}s",
            )
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        output = (completed.stdout or "") + (completed.stderr or "")
        success = completed.returncode == 0
        return ToolResult(
            success=success,
            output=output.strip() or f"(exit {completed.returncode})",
            data={"exit_code": completed.returncode, "command": command},
            error=None if success else f"exit code {completed.returncode}",
        )

    def _is_allowed(self, command: str) -> bool:
        normalized = " ".join(command.split())
        for allowed in self.allowlist:
            allowed_norm = " ".join(allowed.split())
            if normalized == allowed_norm or normalized.startswith(allowed_norm + " "):
                if any(token in normalized for token in (";", "&&", "||", "|", "`", "$(", ">")):
                    if normalized != allowed_norm:
                        return False
                return True
        try:
            parts = shlex.split(normalized)
        except ValueError:
            return False
        rebuilt = " ".join(parts)
        return rebuilt in {" ".join(a.split()) for a in self.allowlist}
