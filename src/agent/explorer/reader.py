"""Selective file reader for important files only.

Reads a bounded amount of text from nominated paths. Never walks or reads
the entire repository.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.explorer.models import ImportantFile
from agent.utils.ignore import IgnoreMatcher

logger = logging.getLogger(__name__)


class ImportantFileReader:
    """Read contents for a small set of important files."""

    def __init__(
        self,
        repo_root: Path,
        *,
        max_bytes: int = 100_000,
        ignore_matcher: IgnoreMatcher | None = None,
    ) -> None:
        """Initialize the reader.

        Args:
            repo_root: Repository root path.
            max_bytes: Maximum bytes to read per file before truncation.
            ignore_matcher: Optional ignore rules (paths outside root are skipped).
        """
        self.repo_root = repo_root.resolve()
        self.max_bytes = max_bytes
        self.ignore_matcher = ignore_matcher or IgnoreMatcher.from_default_patterns()

    def read_files(self, paths_with_roles: list[tuple[str, str]]) -> list[ImportantFile]:
        """Read each relative path and return ImportantFile models.

        Args:
            paths_with_roles: List of ``(relative_path, role)`` pairs.
        """
        results: list[ImportantFile] = []
        for relative_path, role in paths_with_roles:
            important = self.read_one(relative_path, role)
            results.append(important)
        logger.info("Read %d important files", len(results))
        return results

    def read_one(self, relative_path: str, role: str) -> ImportantFile:
        """Read a single important file.

        Returns an ImportantFile with ``content=None`` when unreadable.
        """
        absolute = (self.repo_root / relative_path).resolve()
        if self.ignore_matcher.is_ignored(absolute, repo_root=self.repo_root):
            logger.warning("Refusing to read ignored path: %s", relative_path)
            return ImportantFile(path=relative_path, role=role, content=None)

        try:
            absolute.relative_to(self.repo_root)
        except ValueError:
            logger.warning("Refusing to read path outside repo: %s", relative_path)
            return ImportantFile(path=relative_path, role=role, content=None)

        if not absolute.is_file():
            logger.warning("Important file not found: %s", relative_path)
            return ImportantFile(path=relative_path, role=role, content=None)

        try:
            raw = absolute.read_bytes()
        except OSError:
            logger.warning("Failed to read file: %s", relative_path, exc_info=True)
            return ImportantFile(path=relative_path, role=role, content=None)

        truncated = False
        if len(raw) > self.max_bytes:
            raw = raw[: self.max_bytes]
            truncated = True
            logger.debug("Truncated file %s to %d bytes", relative_path, self.max_bytes)

        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = raw.decode("utf-8", errors="replace")
                logger.debug("Decoded %s with replacement characters", relative_path)
            except Exception:
                logger.warning("Binary or undecodable file skipped: %s", relative_path)
                return ImportantFile(path=relative_path, role=role, content=None, truncated=truncated)

        return ImportantFile(
            path=relative_path,
            role=role,
            content=text,
            truncated=truncated,
        )
