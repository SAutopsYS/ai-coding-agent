"""Prompt builder for the act / edit stage."""

from __future__ import annotations

import json

from agent.explorer.models import RepositorySummary
from agent.llm.client import ChatMessage
from agent.planner.models import ExecutionPlan


def build_act_messages(
    *,
    request: str,
    summary: RepositorySummary,
    plan: ExecutionPlan,
    file_contents: dict[str, str],
    available_tools: list[str] | None = None,
) -> list[ChatMessage]:
    """Build chat messages asking for structured EditInstructions JSON."""
    tools = available_tools or []
    system = """You are a careful coding agent.
Return ONLY valid JSON matching this schema:
{
  "thought": "string",
  "edits": [
    {
      "path": "relative/path",
      "action": "replace" | "write",
      "old_string": "exact text to replace (for replace)",
      "new_string": "replacement text (for replace)",
      "content": "full file content (for write)",
      "reason": "why this edit is needed"
    }
  ],
  "done": true,
  "notes": "string"
}
Rules:
- Never return free-form code outside JSON.
- Prefer action=replace with unique old_string context.
- Preserve existing functionality unless the request requires a change.
- Do not invent files that do not exist unless creating a new file via write.
- Keep edits minimal and focused on the execution plan.
"""

    summary_payload = {
        "project_name": summary.project_name,
        "project_type": summary.project_type,
        "language": summary.detected_language,
        "framework": summary.framework,
        "entry_point": summary.entry_point,
        "technology_stack": summary.technology_stack,
        "important_directories": summary.important_directories,
        "tree": summary.repository_tree_summary,
    }
    plan_payload = plan.model_dump()
    plan_payload.pop("request", None)

    file_blocks: list[str] = []
    for path, content in file_contents.items():
        file_blocks.append(
            f'<<<FILE path="{path}">>>\n{content}\n<<<END_FILE>>>'
        )

    user = (
        f"USER REQUEST:\n{request}\n\n"
        f"REPOSITORY SUMMARY JSON:\n{json.dumps(summary_payload, indent=2)}\n\n"
        f"EXECUTION PLAN JSON:\n{json.dumps(plan_payload, indent=2)}\n\n"
        f"AVAILABLE TOOLS (already used for context gathering): {', '.join(tools)}\n\n"
        f"RELEVANT FILE CONTENTS:\n" + "\n\n".join(file_blocks)
    )

    return [
        ChatMessage(role="system", content=system),
        ChatMessage(role="user", content=user),
    ]
