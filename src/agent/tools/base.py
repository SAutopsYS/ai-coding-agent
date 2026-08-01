"""Tool protocol and registry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Normalized result returned by every tool."""

    success: bool
    output: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@runtime_checkable
class Tool(Protocol):
    """Protocol that concrete tools must satisfy."""

    name: str
    description: str

    def run(self, args: dict[str, Any]) -> ToolResult:
        """Execute the tool and return a normalized ToolResult."""
        ...


class ToolRegistry:
    """Name-to-tool mapping for discovery and dispatch."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool instance by its ``name``."""
        if not getattr(tool, "name", None):
            raise ValueError("Tool must have a non-empty name")
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool registration: {tool.name}")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)

    def get(self, name: str) -> Tool:
        """Return a registered tool or raise KeyError."""
        return self._tools[name]

    def list_tools(self) -> list[dict[str, str]]:
        """Return ``[{name, description}, ...]`` for prompt injection."""
        return [
            {"name": tool.name, "description": tool.description}
            for tool in self._tools.values()
        ]

    def run(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Dispatch a tool call by name."""
        try:
            tool = self.get(name)
        except KeyError:
            return ToolResult(success=False, output="", error=f"Unknown tool: {name}")
        try:
            result = tool.run(args or {})
            logger.info("Tool %s success=%s", name, result.success)
            return result
        except Exception as exc:  # noqa: BLE001 - surface tool failures to agent loop
            logger.exception("Tool %s failed", name)
            return ToolResult(success=False, output="", error=str(exc))
