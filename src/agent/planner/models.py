"""Pydantic models for deterministic execution planning.

These models are produced by heuristic planning (no LLM) from a
RepositorySummary plus a natural-language product request.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RelevantFile(BaseModel):
    """A repository file selected as relevant to the request."""

    path: str = Field(description="Path relative to the repository root.")
    reason: str = Field(description="Why this file is relevant to the request.")
    priority: int = Field(
        default=0,
        description="Higher means more important for implementation (heuristic score).",
    )
    role_hint: str = Field(
        default="unknown",
        description="Inferred role such as model, controller, route, config, entrypoint.",
    )


class PlanStep(BaseModel):
    """One ordered implementation step in the execution plan."""

    id: str
    title: str
    description: str
    files: list[str] = Field(
        default_factory=list,
        description="Files expected to be touched for this step.",
    )
    rationale: str = Field(
        default="",
        description="Why this step is needed given the request and repo shape.",
    )


class Risk(BaseModel):
    """A potential risk associated with implementing the plan."""

    description: str
    severity: Literal["low", "medium", "high"] = "medium"
    mitigation: str = ""


class ValidationItem(BaseModel):
    """A checklist item used to validate the change after implementation."""

    description: str
    category: str = Field(
        default="general",
        description="Category such as api, regression, data, docs.",
    )


class ExecutionPlan(BaseModel):
    """Structured execution plan for satisfying a product request."""

    goal: str
    assumptions: list[str] = Field(default_factory=list)
    relevant_files: list[RelevantFile] = Field(default_factory=list)
    steps: list[PlanStep] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    validation_checklist: list[ValidationItem] = Field(default_factory=list)
    unchanged_functionality: list[str] = Field(
        default_factory=list,
        description="Existing behaviors that must remain unchanged.",
    )
    detected_intents: list[str] = Field(
        default_factory=list,
        description="Heuristic intents inferred from the user request.",
    )
    project_type: str = ""
    request: str = ""
