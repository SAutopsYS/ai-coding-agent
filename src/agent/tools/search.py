"""Search tools: glob by name and content search."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from agent.safety.sandbox import Sandbox
from agent.tools.base import ToolResult
from agent.utils.ignore import IgnoreMatcher

logger = logging.getLogger(__name__)


class GlobFilesTool:
    """Find files matching a glob pattern under the repo root."""

    name: str = "glob_files"
    description: str = "Find files by glob pattern (e.g. **/package.json)."

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
        pattern = str(args.get("pattern") or "")
        if not pattern:
            return ToolResult(success=False, output="", error="pattern is required")
        max_results = int(args.get("max_results") or 200)

        matches: list[str] = []
        for path in self.repo_root.glob(pattern):
            if not path.is_file():
                continue
            if self.ignore_matcher.is_ignored(path, repo_root=self.repo_root):
                continue
            if not self.sandbox.is_inside(path.resolve()):
                continue
            matches.append(self.sandbox.to_relative(path))
            if len(matches) >= max_results:
                break

        matches.sort()
        logger.info("glob_files pattern=%s matches=%d", pattern, len(matches))
        return ToolResult(
            success=True,
            output="\n".join(matches) if matches else "(no matches)",
            data={"matches": matches},
        )


class SearchContentTool:
    """Search file contents for a regex or literal pattern."""

    name: str = "search_content"
    description: str = "Search repository file contents for a pattern."

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
        pattern = str(args.get("pattern") or "")
        if not pattern:
            return ToolResult(success=False, output="", error="pattern is required")
        glob_filter = str(args.get("glob") or "**/*")
        max_results = int(args.get("max_results") or 50)
        case_insensitive = bool(args.get("case_insensitive", True))

        flags = re.IGNORECASE if case_insensitive else 0
        try:
            regex = re.compile(pattern, flags)
        except re.error as exc:
            return ToolResult(success=False, output="", error=f"Invalid regex: {exc}")

        hits: list[dict[str, Any]] = []
        lines_out: list[str] = []

        for path in self.repo_root.glob(glob_filter):
            if not path.is_file():
                continue
            if self.ignore_matcher.is_ignored(path, repo_root=self.repo_root):
                continue
            if not self.sandbox.is_inside(path.resolve()):
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for lineno, line in enumerate(text.splitlines(), start=1):
                if regex.search(line):
                    rel = self.sandbox.to_relative(path)
                    hit = {"path": rel, "line": lineno, "text": line.strip()[:240]}
                    hits.append(hit)
                    lines_out.append(f"{rel}:{lineno}: {hit['text']}")
                    if len(hits) >= max_results:
                        output = "\n".join(lines_out)
                        return ToolResult(
                            success=True,
                            output=output,
                            data={"matches": hits, "truncated": True},
                        )

        output = "\n".join(lines_out) if lines_out else "(no matches)"
        logger.info("search_content pattern=%s hits=%d", pattern, len(hits))
        return ToolResult(success=True, output=output, data={"matches": hits})
