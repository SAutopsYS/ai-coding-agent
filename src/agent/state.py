"""In-memory workspace state shared across agent stages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent.explorer.models import RepositorySummary
    from agent.llm.schemas import FinalReport
    from agent.planner.models import ExecutionPlan
    from agent.verify.verifier import VerificationResult


@dataclass
class PlanStep:
    """A single step in the execution plan (runtime tracking)."""

    id: str
    description: str
    files: list[str] = field(default_factory=list)
    done: bool = False


@dataclass
class ToolTranscriptEntry:
    """Record of one tool invocation and its result."""

    tool_name: str
    args: dict[str, Any]
    result_preview: str
    success: bool = True


@dataclass
class FileChange:
    """Metadata about a file modified by the agent."""

    path: str
    change_type: str  # created | modified
    summary: str = ""


@dataclass
class WorkspaceState:
    """Mutable state for a single agent run."""

    repo_root: Path
    request: str
    file_tree_summary: str = ""
    stack_summary: str = ""
    entrypoint: str = ""
    relevant_files: list[str] = field(default_factory=list)
    file_excerpts: dict[str, str] = field(default_factory=dict)
    plan_steps: list[PlanStep] = field(default_factory=list)
    invariants: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    tool_transcript: list[ToolTranscriptEntry] = field(default_factory=list)
    files_changed: list[FileChange] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    verify_notes: str = ""
    final_summary: str = ""
    step_count: int = 0
    repository_summary: RepositorySummary | None = None
    execution_plan: ExecutionPlan | None = None
    verification: VerificationResult | None = None
    final_report: FinalReport | None = None

    def append_tool_result(
        self,
        tool_name: str,
        args: dict[str, Any],
        preview: str,
        *,
        success: bool,
    ) -> None:
        """Append a truncated tool transcript entry."""
        self.tool_transcript.append(
            ToolTranscriptEntry(
                tool_name=tool_name,
                args=args,
                result_preview=preview[:1000],
                success=success,
            )
        )

    def record_file_change(self, path: str, change_type: str, summary: str = "") -> None:
        """Track a file mutation, replacing prior entry for the same path."""
        self.files_changed = [item for item in self.files_changed if item.path != path]
        self.files_changed.append(
            FileChange(path=path, change_type=change_type, summary=summary)
        )

    def to_report_dict(self) -> dict[str, Any]:
        """Serialize the final report / key state for JSON output."""
        if self.final_report is not None:
            return self.final_report.model_dump()
        return {
            "overview": self.final_summary,
            "files_modified": [
                {"path": f.path, "why": f.summary} for f in self.files_changed
            ],
            "errors": self.errors,
            "verify_notes": self.verify_notes,
        }
