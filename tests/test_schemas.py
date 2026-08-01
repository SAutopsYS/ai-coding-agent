"""Tests for LLM Pydantic schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent.llm.schemas import ActTurn, EditInstructions, ExploreResult, FileEdit, Summary


def test_explore_result_parses() -> None:
    result = ExploreResult(
        stack="Node.js + Express + MongoDB",
        entrypoint="server.js",
        structure_summary="MVC-style app/ with routes, controllers, models.",
        key_dirs=["app", "config"],
    )
    assert result.entrypoint == "server.js"


def test_file_edit_replace_requires_strings() -> None:
    with pytest.raises(ValidationError):
        FileEdit(path="a.js", action="replace", reason="x")


def test_edit_instructions_parse() -> None:
    payload = EditInstructions.model_validate(
        {
            "thought": "add tags",
            "edits": [
                {
                    "path": "app/models/note.model.js",
                    "action": "replace",
                    "old_string": "content: String",
                    "new_string": "content: String,\n    tags: [String]",
                    "reason": "organize",
                }
            ],
            "done": True,
        }
    )
    assert payload.edits[0].path.endswith("note.model.js")


def test_act_turn_done() -> None:
    turn = ActTurn(thought="finished", action="done")
    assert turn.action == "done"


def test_summary_parses() -> None:
    summary = Summary(
        overview="Summary test payload",
        files_changed=[],
        behavior_added=[],
        preserved=["CRUD"],
        follow_ups=[],
    )
    assert "CRUD" in summary.preserved
