"""Tests for LLM JSON parsing helpers."""

from __future__ import annotations

from agent.llm.parse import extract_json_object, parse_model
from agent.llm.schemas import EditInstructions


def test_extract_fenced_json() -> None:
    text = """Here you go:
```json
{"thought": "x", "edits": [], "done": true}
```
"""
    assert '"done": true' in extract_json_object(text)


def test_parse_edit_instructions() -> None:
    raw = '{"thought":"ok","edits":[],"done":true,"notes":""}'
    model = parse_model(raw, EditInstructions)
    assert model.done is True
