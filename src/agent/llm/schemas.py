"""Pydantic models for structured LLM outputs."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class ExploreResult(BaseModel):
    """Optional LLM explore output schema."""

    stack: str
    entrypoint: str
    structure_summary: str
    key_dirs: list[str] = Field(default_factory=list)


class RelevantFile(BaseModel):
    """Optional LLM-ranked relevant file."""

    path: str
    reason: str


class RelevantFiles(BaseModel):
    """Collection of LLM-ranked relevant files."""

    files: list[RelevantFile] = Field(default_factory=list)


class FileEdit(BaseModel):
    """One structured file mutation instruction."""

    path: str
    action: Literal["replace", "write"] = "replace"
    old_string: str | None = None
    new_string: str | None = None
    content: str | None = None
    reason: str = ""

    @model_validator(mode="after")
    def validate_payload(self) -> FileEdit:
        """Ensure replace/write actions include the required fields."""
        if self.action == "replace":
            if not self.old_string or self.new_string is None:
                raise ValueError("replace requires old_string and new_string")
        if self.action == "write":
            if self.content is None:
                raise ValueError("write requires content")
        return self


class EditInstructions(BaseModel):
    """Primary structured response from the coding LLM."""

    thought: str = ""
    edits: list[FileEdit] = Field(default_factory=list)
    done: bool = False
    notes: str = ""


class ToolCallAction(BaseModel):
    """Structured tool invocation requested by the LLM."""

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class EditAction(BaseModel):
    """Single-file edit action used by the optional ActTurn format."""

    path: str
    old_string: str | None = None
    new_string: str | None = None
    patch: str | None = None
    reason: str = ""


class ActTurn(BaseModel):
    """Optional iterative act-turn format (tool, edit, or done)."""

    thought: str = ""
    action: Literal["tool", "edit", "edits", "done"] = "done"
    tool: ToolCallAction | None = None
    edit: EditAction | None = None
    edits: list[FileEdit] = Field(default_factory=list)


class VerificationAssessment(BaseModel):
    """Structured verification judgment from heuristics/LLM."""

    syntax_ok: bool = True
    feature_implemented: bool = False
    existing_preserved: bool = True
    notes: list[str] = Field(default_factory=list)
    checks: list[str] = Field(default_factory=list)


class FinalReport(BaseModel):
    """Final agent report returned to the user."""

    overview: str
    files_modified: list[dict[str, str]] = Field(default_factory=list)
    summary_of_changes: list[str] = Field(default_factory=list)
    validation_results: list[dict[str, str]] = Field(default_factory=list)
    preserved_functionality: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
    success: bool = False


class Summary(BaseModel):
    """Compact summary schema."""

    overview: str
    files_changed: list[str] = Field(default_factory=list)
    behavior_added: list[str] = Field(default_factory=list)
    preserved: list[str] = Field(default_factory=list)
    follow_ups: list[str] = Field(default_factory=list)
