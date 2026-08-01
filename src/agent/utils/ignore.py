"""Ignore rules for exploration and search.

Skips common dependency, VCS, and build directories so repository scans stay
fast and relevant across languages.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_IGNORED_DIR_NAMES: frozenset[str] = frozenset(
    {
        ".git",
        "node_modules",
        "dist",
        "build",
        "__pycache__",
        ".venv",
    }
)


@dataclass
class IgnoreMatcher:
    """Decides whether a path should be skipped during explore/search."""

    ignored_dir_names: frozenset[str] = field(default_factory=lambda: DEFAULT_IGNORED_DIR_NAMES)
    patterns: list[str] = field(default_factory=list)

    @classmethod
    def from_default_patterns(cls) -> IgnoreMatcher:
        """Construct with the built-in ignore directory set."""
        return cls(
            ignored_dir_names=DEFAULT_IGNORED_DIR_NAMES,
            patterns=[
                "**/.git/**",
                "**/node_modules/**",
                "**/dist/**",
                "**/build/**",
                "**/__pycache__/**",
                "**/.venv/**",
            ],
        )

    def is_ignored(self, path: Path, *, repo_root: Path | None = None) -> bool:
        """Return True if ``path`` should be excluded from scanning.

        A path is ignored when any of its path segments (relative to
        ``repo_root`` when provided) matches a known ignored directory name.

        Args:
            path: File or directory path to test.
            repo_root: Optional root used to compute a relative path first.
        """
        try:
            if repo_root is not None:
                try:
                    relative = path.resolve().relative_to(repo_root.resolve())
                except ValueError:
                    logger.debug("Path outside repo root treated as ignored: %s", path)
                    return True
            else:
                relative = path
        except OSError:
            logger.debug("Failed to resolve path for ignore check: %s", path, exc_info=True)
            return True

        for part in relative.parts:
            if part in self.ignored_dir_names:
                return True
        return False
