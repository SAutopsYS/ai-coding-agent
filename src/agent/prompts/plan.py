"""Prompt builder for an optional LLM plan stage."""

from __future__ import annotations

from agent.llm.client import ChatMessage
from agent.state import WorkspaceState


def build_plan_messages(state: WorkspaceState) -> list[ChatMessage]:
    """Build planning messages from workspace state."""
    system = (
        "Create a short implementation plan as JSON with goal, steps, "
        "invariants, and risks. Do not edit files."
    )
    user = (
        f"Request: {state.request}\n"
        f"Entrypoint: {state.entrypoint}\n"
        f"Relevant files: {', '.join(state.relevant_files)}\n"
        f"Excerpts:\n{state.file_excerpts}\n"
    )
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
