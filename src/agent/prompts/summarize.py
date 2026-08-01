"""Prompt builder for the summarize stage (also used for heuristic fallback)."""

from __future__ import annotations

import json

from agent.llm.client import ChatMessage
from agent.state import FileChange, WorkspaceState


def build_summarize_messages(state: WorkspaceState) -> list[ChatMessage]:
    """Build chat messages for a final change summary."""
    system = """You summarize code changes made by an agent.
Return ONLY JSON:
{
  "overview": "string",
  "files_changed": ["path"],
  "behavior_added": ["string"],
  "preserved": ["string"],
  "follow_ups": ["string"]
}
"""
    payload = {
        "request": state.request,
        "files_changed": [
            {"path": f.path, "change_type": f.change_type, "summary": f.summary}
            for f in state.files_changed
        ],
        "verify_notes": state.verify_notes,
        "errors": state.errors,
        "invariants": state.invariants,
    }
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=json.dumps(payload, indent=2)),
    ]


def build_heuristic_summary(
    *,
    request: str,
    files_changed: list[FileChange],
    validation_notes: list[str],
    preserved: list[str],
) -> dict[str, object]:
    """Deterministic summary when LLM summarization is unavailable/unneeded."""
    return {
        "overview": (
            f"Applied structured edits for request: {request.strip()}"
            if files_changed
            else f"No files modified for request: {request.strip()}"
        ),
        "files_changed": [f.path for f in files_changed],
        "behavior_added": [f.summary for f in files_changed if f.summary],
        "preserved": preserved,
        "follow_ups": validation_notes,
    }
