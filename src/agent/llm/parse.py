"""Parse structured JSON payloads from LLM text responses."""

from __future__ import annotations

import json
import logging
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


def extract_json_object(text: str) -> str:
    """Extract the first JSON object from model output (supports fenced blocks)."""
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, flags=re.DOTALL)
    if fenced:
        return fenced.group(1)

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in LLM response")
    return stripped[start : end + 1]


def parse_model(text: str, model_type: type[T]) -> T:
    """Parse and validate LLM text into a Pydantic model."""
    payload = extract_json_object(text)
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        logger.error("Failed to decode LLM JSON: %s", exc)
        raise
    try:
        return model_type.model_validate(data)
    except ValidationError:
        logger.error("LLM JSON failed schema validation for %s", model_type.__name__)
        raise
