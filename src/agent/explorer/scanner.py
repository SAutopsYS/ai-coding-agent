"""Recursive repository scanner.

Walks a repository root, skips ignored directories, and builds an in-memory
RepositoryMap (file/directory lists + nested tree).
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.explorer.models import RepoNode, RepositoryMap
from agent.utils.ignore import IgnoreMatcher

logger = logging.getLogger(__name__)


class RepositoryScanner:
    """Scan a repository into an in-memory map."""

    def __init__(
        self,
        repo_root: Path,
        *,
        ignore_matcher: IgnoreMatcher | None = None,
        max_entries: int = 50_000,
    ) -> None:
        """Initialize the scanner.

        Args:
            repo_root: Absolute or relative path to the repository root.
            ignore_matcher: Rules for skipped directories.
            max_entries: Safety cap on total files + directories discovered.
        """
        self.repo_root = repo_root.resolve()
        self.ignore_matcher = ignore_matcher or IgnoreMatcher.from_default_patterns()
        self.max_entries = max_entries

    def scan(self) -> RepositoryMap:
        """Recursively scan the repository and return a RepositoryMap.

        Raises:
            FileNotFoundError: If ``repo_root`` does not exist.
            NotADirectoryError: If ``repo_root`` is not a directory.
        """
        if not self.repo_root.exists():
            raise FileNotFoundError(f"Repository path does not exist: {self.repo_root}")
        if not self.repo_root.is_dir():
            raise NotADirectoryError(f"Repository path is not a directory: {self.repo_root}")

        logger.info("Scanning repository at %s", self.repo_root)

        files: list[str] = []
        directories: list[str] = []
        tree = RepoNode(name=self.repo_root.name, path=".", kind="directory", children=[])
        self._walk(self.repo_root, tree, files, directories)

        files.sort()
        directories.sort()

        repo_map = RepositoryMap(
            root=str(self.repo_root),
            files=files,
            directories=directories,
            tree=tree,
            file_count=len(files),
            directory_count=len(directories),
        )
        logger.info(
            "Scan complete: %d files, %d directories",
            repo_map.file_count,
            repo_map.directory_count,
        )
        return repo_map

    def _walk(
        self,
        absolute_dir: Path,
        node: RepoNode,
        files: list[str],
        directories: list[str],
    ) -> None:
        """Depth-first walk into ``absolute_dir``, mutating ``node`` and lists."""
        try:
            entries = sorted(absolute_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except OSError:
            logger.warning("Cannot list directory: %s", absolute_dir, exc_info=True)
            return

        for entry in entries:
            if len(files) + len(directories) >= self.max_entries:
                logger.warning(
                    "Reached max_entries=%d; stopping scan early",
                    self.max_entries,
                )
                return

            if self.ignore_matcher.is_ignored(entry, repo_root=self.repo_root):
                logger.debug("Ignoring path: %s", entry)
                continue

            try:
                relative = entry.relative_to(self.repo_root).as_posix()
            except ValueError:
                logger.debug("Skipping non-relative path: %s", entry)
                continue

            if entry.is_dir():
                directories.append(relative)
                child = RepoNode(name=entry.name, path=relative, kind="directory", children=[])
                node.children.append(child)
                self._walk(entry, child, files, directories)
            elif entry.is_file():
                files.append(relative)
                node.children.append(
                    RepoNode(name=entry.name, path=relative, kind="file", children=[])
                )
