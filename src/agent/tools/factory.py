"""Build a fully populated ToolRegistry for a repository."""

from __future__ import annotations

from pathlib import Path

from agent.config_loader import AgentConfig
from agent.safety.limits import Limits
from agent.safety.sandbox import Sandbox
from agent.tools.base import ToolRegistry
from agent.tools.filesystem import (
    ApplyPatchTool,
    ListDirectoryTool,
    ReadFileTool,
    WriteFileTool,
)
from agent.tools.git_tools import GitCommitTool, GitDiffTool, GitLogTool, GitStatusTool
from agent.tools.search import GlobFilesTool, SearchContentTool
from agent.tools.shell import ShellTool
from agent.utils.ignore import IgnoreMatcher


def build_tool_registry(
    repo_root: Path,
    config: AgentConfig,
    *,
    dry_run: bool = False,
) -> ToolRegistry:
    """Create and register all runtime tools."""
    root = repo_root.resolve()
    sandbox = Sandbox(root)
    ignore = IgnoreMatcher.from_default_patterns()
    limits = Limits(
        max_steps=config.max_steps,
        max_files_touched=config.max_files_touched,
        max_read_bytes=config.max_read_bytes,
        max_write_bytes=config.max_write_bytes,
        max_repair_attempts=config.max_repair_attempts,
    )

    registry = ToolRegistry()
    registry.register(ListDirectoryTool(root, sandbox=sandbox, ignore_matcher=ignore))
    registry.register(ReadFileTool(root, sandbox=sandbox, limits=limits))
    registry.register(WriteFileTool(root, sandbox=sandbox, limits=limits, dry_run=dry_run))
    registry.register(ApplyPatchTool(root, sandbox=sandbox, limits=limits, dry_run=dry_run))
    registry.register(GlobFilesTool(root, sandbox=sandbox, ignore_matcher=ignore))
    registry.register(SearchContentTool(root, sandbox=sandbox, ignore_matcher=ignore))
    registry.register(
        ShellTool(
            root,
            allowlist=config.shell_allowlist,
            timeout_seconds=config.shell_timeout_seconds,
            enabled=config.shell_enabled,
        )
    )
    registry.register(GitStatusTool(root))
    registry.register(GitDiffTool(root))
    registry.register(GitLogTool(root))
    registry.register(GitCommitTool(root, allow_commit=config.git_allow_commit))
    return registry
