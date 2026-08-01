"""Repository Explorer facade.

Coordinates scanning, important-file detection, selective reading, and
heuristic metadata inference. Produces a RepositorySummary without LLM use.
"""

from __future__ import annotations

import logging
from pathlib import Path

from agent.explorer.detector import ProjectDetector
from agent.explorer.models import ImportantFile, RepositorySummary
from agent.explorer.reader import ImportantFileReader
from agent.explorer.scanner import RepositoryScanner
from agent.explorer.tree import format_tree_summary
from agent.utils.ignore import IgnoreMatcher

logger = logging.getLogger(__name__)


class RepositoryExplorer:
    """Explore an unknown software repository and return a structured summary."""

    def __init__(
        self,
        repo_path: str | Path,
        *,
        ignore_matcher: IgnoreMatcher | None = None,
        max_file_bytes: int = 100_000,
        max_scan_entries: int = 50_000,
    ) -> None:
        """Create an explorer bound to a repository path.

        Args:
            repo_path: Path to the target repository root.
            ignore_matcher: Optional custom ignore rules.
            max_file_bytes: Per-file read limit for important files.
            max_scan_entries: Safety cap for scan breadth.
        """
        self.repo_root = Path(repo_path).expanduser().resolve()
        self.ignore_matcher = ignore_matcher or IgnoreMatcher.from_default_patterns()
        self.max_file_bytes = max_file_bytes
        self.max_scan_entries = max_scan_entries

    def explore(self) -> RepositorySummary:
        """Run a full exploration and return a RepositorySummary.

        Steps:
            1. Recursively scan (honoring ignore rules) into a RepositoryMap.
            2. Detect important files and directories.
            3. Read only those important files.
            4. Infer language, project type, framework, entry point, and stack.
            5. Build a compact tree summary string.
        """
        logger.info("Starting repository exploration for %s", self.repo_root)

        scanner = RepositoryScanner(
            self.repo_root,
            ignore_matcher=self.ignore_matcher,
            max_entries=self.max_scan_entries,
        )
        repo_map = scanner.scan()

        detector = ProjectDetector(self.repo_root)
        important_paths = detector.detect_important_file_paths(repo_map)
        important_directories = detector.detect_important_directories(repo_map)

        reader = ImportantFileReader(
            self.repo_root,
            max_bytes=self.max_file_bytes,
            ignore_matcher=self.ignore_matcher,
        )
        important_files = reader.read_files(important_paths)
        file_contents = {
            item.path: item.content
            for item in important_files
            if item.content is not None
        }

        language, project_type = detector.detect_language_and_type(repo_map, file_contents)
        project_name = detector.detect_project_name(repo_map, file_contents)
        framework = detector.detect_framework(project_type, file_contents, repo_map)
        entry_point = detector.detect_entry_point(project_type, repo_map, file_contents)

        important_files = self._ensure_entry_point_file(
            important_files,
            entry_point,
            reader,
        )
        file_contents = {
            item.path: item.content
            for item in important_files
            if item.content is not None
        }

        technology_stack = detector.build_technology_stack(
            language=language,
            project_type=project_type,
            framework=framework,
            important_files=important_files,
            file_contents=file_contents,
        )
        tree_summary = format_tree_summary(repo_map)

        summary = RepositorySummary(
            project_name=project_name,
            detected_language=language,
            project_type=project_type,
            framework=framework,
            entry_point=entry_point,
            important_directories=important_directories,
            important_files=important_files,
            technology_stack=technology_stack,
            repository_tree_summary=tree_summary,
            repository_map=repo_map,
        )

        logger.info(
            "Exploration finished: name=%s type=%s language=%s framework=%s entry=%s",
            summary.project_name,
            summary.project_type,
            summary.detected_language,
            summary.framework,
            summary.entry_point,
        )
        return summary

    def _ensure_entry_point_file(
        self,
        important_files: list[ImportantFile],
        entry_point: str | None,
        reader: ImportantFileReader,
    ) -> list[ImportantFile]:
        """If an entry point was inferred and not already loaded, read it once."""
        if not entry_point:
            return important_files

        existing_paths = {item.path for item in important_files}
        if entry_point in existing_paths:
            updated: list[ImportantFile] = []
            for item in important_files:
                if item.path == entry_point and item.role != "entrypoint":
                    updated.append(
                        item.model_copy(update={"role": f"{item.role},entrypoint"})
                    )
                else:
                    updated.append(item)
            return updated

        logger.info("Reading inferred entry point: %s", entry_point)
        entry_file = reader.read_one(entry_point, "entrypoint")
        return [*important_files, entry_file]
