"""Prompt builder for an optional LLM explore stage."""

from __future__ import annotations

from agent.llm.client import ChatMessage
from agent.state import WorkspaceState


def build_explore_messages(state: WorkspaceState) -> list[ChatMessage]:
    """Build explore messages from workspace state excerpts."""
    system = (
        "Summarize the repository stack and layout. Return JSON with keys "
        "stack, entrypoint, structure_summary, key_dirs."
    )
    user = (
        f"Request: {state.request}\n"
        f"Tree:\n{state.file_tree_summary}\n"
        f"Stack hint:\n{state.stack_summary}\n"
    )
    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
