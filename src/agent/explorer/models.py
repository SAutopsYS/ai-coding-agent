"""Pydantic models for repository exploration results.

These models are heuristic (no LLM). They describe what the explorer scanned
and inferred about a target repository.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RepoNode(BaseModel):
    """One node in the in-memory repository tree."""

    name: str
    path: str = Field(description="Path relative to the repository root.")
    kind: Literal["file", "directory"]
    children: list[RepoNode] = Field(default_factory=list)


class RepositoryMap(BaseModel):
    """In-memory map of a scanned repository (structure only, no file bodies)."""

    root: str = Field(description="Absolute path to the repository root.")
    files: list[str] = Field(
        default_factory=list,
        description="All discovered file paths relative to the root.",
    )
    directories: list[str] = Field(
        default_factory=list,
        description="All discovered directory paths relative to the root.",
    )
    tree: RepoNode | None = Field(
        default=None,
        description="Nested tree representation of the repository.",
    )
    file_count: int = 0
    directory_count: int = 0


class ImportantFile(BaseModel):
    """An important file identified during exploration."""

    path: str = Field(description="Path relative to the repository root.")
    role: str = Field(
        description="Role hint, e.g. manifest, readme, entrypoint, container, config.",
    )
    content: str | None = Field(
        default=None,
        description="File text when read; None if not read or unreadable.",
    )
    truncated: bool = Field(
        default=False,
        description="True when content was truncated due to size limits.",
    )


class RepositorySummary(BaseModel):
    """Structured summary produced by RepositoryExplorer."""

    project_name: str
    detected_language: str
    project_type: str = Field(
        description="High-level project family, e.g. Node.js, Python, Java, Go, Rust.",
    )
    framework: str | None = None
    entry_point: str | None = None
    important_directories: list[str] = Field(default_factory=list)
    important_files: list[ImportantFile] = Field(default_factory=list)
    technology_stack: list[str] = Field(default_factory=list)
    repository_tree_summary: str = Field(
        default="",
        description="Compact textual tree for logging and later LLM context.",
    )
    repository_map: RepositoryMap | None = Field(
        default=None,
        description="Full in-memory map from the scan.",
    )
