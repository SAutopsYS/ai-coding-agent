"""Load YAML configuration and environment overrides for the agent."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"


def _load_dotenv() -> None:
    """Load project-root `.env` if present. Skip silently when missing or unreadable."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    env_path = DEFAULT_CONFIG_PATH.parents[1] / ".env"
    if not env_path.is_file():
        return
    try:
        load_dotenv(dotenv_path=env_path, override=False, encoding="utf-8")
        logger.info("Loaded environment file %s", env_path)
    except UnicodeDecodeError:
        logger.warning("Skipping .env with invalid UTF-8 encoding: %s", env_path)
    except OSError as exc:
        logger.warning("Could not load .env (%s): %s", env_path, exc)


@dataclass
class AgentConfig:
    """Normalized runtime configuration."""

    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.6-flash"
    llm_temperature: float = 0.2
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    llm_max_retries: int = 3
    max_steps: int = 30
    max_files_touched: int = 20
    max_read_bytes: int = 200_000
    max_write_bytes: int = 200_000
    max_repair_attempts: int = 1
    shell_enabled: bool = False
    shell_timeout_seconds: int = 60
    shell_allowlist: list[str] = field(
        default_factory=lambda: ["npm test", "npm install", "npm run lint", "pytest"]
    )
    git_allow_commit: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


def load_config(path: Path | None = None) -> AgentConfig:
    """Load config from YAML and overlay environment variables."""
    _load_dotenv()
    config_path = path or DEFAULT_CONFIG_PATH
    data: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config root must be a mapping: {config_path}")
        data = loaded
        logger.info("Loaded config from %s", config_path)
    else:
        logger.warning("Config file not found at %s; using defaults", config_path)

    llm = data.get("llm") or {}
    limits = data.get("limits") or {}
    shell = data.get("shell") or {}
    git = data.get("git") or {}

    provider = (
        os.getenv("AGENT_LLM_PROVIDER")
        or str(llm.get("provider") or "gemini")
    ).strip().lower()
    api_key = os.getenv("AGENT_LLM_API_KEY") or llm.get("api_key")
    base_url = os.getenv("AGENT_LLM_BASE_URL") or llm.get("base_url")
    default_model = "gemini-3.6-flash" if provider in {"gemini", "google"} else "gpt-4o-mini"
    model = os.getenv("AGENT_LLM_MODEL") or llm.get("model") or default_model

    allowlist = list(shell.get("allowlist") or [])
    if "pytest" not in allowlist:
        allowlist.append("pytest")

    logger.info(
        "Resolved LLM config provider=%s model=%s api_key_set=%s",
        provider,
        model,
        bool(api_key),
    )

    return AgentConfig(
        llm_provider=provider,
        llm_model=str(model),
        llm_temperature=float(llm.get("temperature", 0.2)),
        llm_api_key=str(api_key) if api_key else None,
        llm_base_url=str(base_url) if base_url else None,
        llm_max_retries=int(llm.get("max_retries", 3)),
        max_steps=int(limits.get("max_steps", 30)),
        max_files_touched=int(limits.get("max_files_touched", 20)),
        max_read_bytes=int(limits.get("max_read_bytes", 200_000)),
        max_write_bytes=int(limits.get("max_write_bytes", 200_000)),
        max_repair_attempts=int(limits.get("max_repair_attempts", 1)),
        shell_enabled=bool(shell.get("enabled", False)),
        shell_timeout_seconds=int(shell.get("timeout_seconds", 60)),
        shell_allowlist=allowlist,
        git_allow_commit=bool(git.get("allow_commit", False)),
        raw=data,
    )
