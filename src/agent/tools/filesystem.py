"""Filesystem tools: list, read, write, apply patch (search-replace)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.safety.limits import LimitExceeded, Limits
from agent.safety.sandbox import Sandbox
from agent.tools.base import ToolResult
from agent.utils.ignore import IgnoreMatcher

logger = logging.getLogger(__name__)


class ListDirectoryTool:
    """List directory entries (depth-limited)."""

    name: str = "list_directory"
    description: str = "List files and directories under a path relative to the repo root."

    def __init__(
        self,
        repo_root: Path,
        *,
        sandbox: Sandbox | None = None,
        ignore_matcher: IgnoreMatcher | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.sandbox = sandbox or Sandbox(self.repo_root)
        self.ignore_matcher = ignore_matcher or IgnoreMatcher.from_default_patterns()

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        relative = str(args.get("path") or ".")
        max_depth = int(args.get("max_depth") or 2)
        try:
            root = self.sandbox.resolve(relative)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        if not root.exists():
            return ToolResult(success=False, output="", error=f"Path not found: {relative}")
        if not root.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {relative}")

        lines: list[str] = []
        self._walk(root, depth=0, max_depth=max_depth, lines=lines)
        output = "\n".join(lines) if lines else "(empty)"
        return ToolResult(success=True, output=output, data={"entries": lines})

    def _walk(self, directory: Path, *, depth: int, max_depth: int, lines: list[str]) -> None:
        if depth > max_depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError as exc:
            lines.append(f"! cannot list {directory}: {exc}")
            return
        for entry in entries:
            if self.ignore_matcher.is_ignored(entry, repo_root=self.repo_root):
                continue
            rel = self.sandbox.to_relative(entry)
            suffix = "/" if entry.is_dir() else ""
            lines.append(f"{rel}{suffix}")
            if entry.is_dir() and depth < max_depth:
                self._walk(entry, depth=depth + 1, max_depth=max_depth, lines=lines)


class ReadFileTool:
    """Read file contents with optional line range and size cap."""

    name: str = "read_file"
    description: str = "Read a text file from the repository (optional start/end lines)."

    def __init__(
        self,
        repo_root: Path,
        *,
        sandbox: Sandbox | None = None,
        limits: Limits | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.sandbox = sandbox or Sandbox(self.repo_root)
        self.limits = limits or Limits()

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        relative = str(args.get("path") or "")
        if not relative:
            return ToolResult(success=False, output="", error="path is required")
        try:
            absolute = self.sandbox.resolve(relative)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        if not absolute.is_file():
            return ToolResult(success=False, output="", error=f"File not found: {relative}")

        try:
            raw = absolute.read_bytes()
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        truncated = False
        if len(raw) > self.limits.max_read_bytes:
            raw = raw[: self.limits.max_read_bytes]
            truncated = True

        text = raw.decode("utf-8", errors="replace")
        start = args.get("start_line")
        end = args.get("end_line")
        if start is not None or end is not None:
            lines = text.splitlines()
            s = max(int(start or 1) - 1, 0)
            e = int(end) if end is not None else len(lines)
            text = "\n".join(lines[s:e])

        return ToolResult(
            success=True,
            output=text,
            data={"path": relative, "truncated": truncated},
        )


class WriteFileTool:
    """Create or overwrite a file."""

    name: str = "write_file"
    description: str = "Write full contents to a file path under the repo root."

    def __init__(
        self,
        repo_root: Path,
        *,
        sandbox: Sandbox | None = None,
        limits: Limits | None = None,
        dry_run: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.sandbox = sandbox or Sandbox(self.repo_root)
        self.limits = limits or Limits()
        self.dry_run = dry_run

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        relative = str(args.get("path") or "")
        content = args.get("content")
        if not relative:
            return ToolResult(success=False, output="", error="path is required")
        if content is None:
            return ToolResult(success=False, output="", error="content is required")
        content_str = str(content)
        try:
            self.limits.check_write_size(len(content_str.encode("utf-8")))
        except LimitExceeded as exc:
            return ToolResult(success=False, output="", error=str(exc))

        try:
            absolute = self.sandbox.resolve(relative)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        created = not absolute.exists()
        if self.dry_run:
            return ToolResult(
                success=True,
                output=f"dry-run write {relative} ({len(content_str)} bytes)",
                data={"path": relative, "created": created, "dry_run": True},
            )

        try:
            absolute.parent.mkdir(parents=True, exist_ok=True)
            absolute.write_text(content_str, encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        logger.info("Wrote file %s created=%s", relative, created)
        return ToolResult(
            success=True,
            output=f"Wrote {relative}",
            data={"path": relative, "created": created},
        )


class ApplyPatchTool:
    """Apply a surgical search-replace edit."""

    name: str = "apply_patch"
    description: str = "Apply a search-replace edit to an existing file."

    def __init__(
        self,
        repo_root: Path,
        *,
        sandbox: Sandbox | None = None,
        limits: Limits | None = None,
        dry_run: bool = False,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.sandbox = sandbox or Sandbox(self.repo_root)
        self.limits = limits or Limits()
        self.dry_run = dry_run

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool with the provided argument map."""
        relative = str(args.get("path") or "")
        old = args.get("old_string")
        new = args.get("new_string")
        if not relative:
            return ToolResult(success=False, output="", error="path is required")
        if old is None or new is None:
            return ToolResult(
                success=False,
                output="",
                error="old_string and new_string are required",
            )
        old_s = str(old)
        new_s = str(new)
        if not old_s:
            return ToolResult(success=False, output="", error="old_string must be non-empty")

        try:
            absolute = self.sandbox.resolve(relative)
        except ValueError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        if not absolute.is_file():
            return ToolResult(success=False, output="", error=f"File not found: {relative}")

        try:
            original = absolute.read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        count = original.count(old_s)
        if count == 0:
            return ToolResult(
                success=False,
                output="",
                error="old_string not found in file (context mismatch)",
                data={"path": relative},
            )
        if count > 1:
            return ToolResult(
                success=False,
                output="",
                error=f"old_string matched {count} times; require a unique match",
                data={"path": relative},
            )

        updated = original.replace(old_s, new_s, 1)
        try:
            self.limits.check_write_size(len(updated.encode("utf-8")))
        except LimitExceeded as exc:
            return ToolResult(success=False, output="", error=str(exc))

        if self.dry_run:
            return ToolResult(
                success=True,
                output=f"dry-run patch {relative}",
                data={"path": relative, "dry_run": True},
            )

        try:
            absolute.write_text(updated, encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))

        logger.info("Patched file %s", relative)
        return ToolResult(
            success=True,
            output=f"Patched {relative}",
            data={"path": relative, "changed": original != updated},
        )
