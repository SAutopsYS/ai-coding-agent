"""Command-line interface for the AI Coding Agent."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Sequence

from agent.config_loader import load_config
from agent.orchestrator import Orchestrator, OrchestratorConfig
from agent.state import WorkspaceState


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="ai-coding-agent",
        description="Explore a repository and apply product-request changes via an AI agent.",
    )
    parser.add_argument(
        "--repo",
        type=Path,
        required=True,
        help="Absolute or relative path to the target repository.",
    )
    parser.add_argument(
        "--request",
        type=str,
        required=False,
        help="Natural-language product request.",
    )
    parser.add_argument(
        "--request-file",
        type=Path,
        required=False,
        help="Path to a file containing the product request.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=False,
        help="Optional path to YAML config (defaults to config/default.yaml).",
    )
    parser.add_argument(
        "--provider",
        type=str,
        required=False,
        help="LLM provider override: openai | anthropic | gemini | mock",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and generate edits without writing files.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        required=False,
        help="Override max act steps.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        required=False,
        help="Optional path to write the final JSON report.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """Configure root logging for the CLI process."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def _resolve_request(args: argparse.Namespace) -> str:
    """Resolve the product request from --request or --request-file."""
    if args.request and args.request.strip():
        return args.request.strip()
    if args.request_file:
        return Path(args.request_file).read_text(encoding="utf-8").strip()
    raise SystemExit("Provide --request or --request-file")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the agent CLI and return a process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    _configure_logging(bool(args.verbose))

    repo = Path(args.repo).expanduser().resolve()
    if not repo.is_dir():
        print(f"Repository not found: {repo}", file=sys.stderr)
        return 2

    request = _resolve_request(args)
    config = load_config(args.config)
    if args.provider:
        config.llm_provider = args.provider.strip().lower()
    if args.max_steps:
        config.max_steps = int(args.max_steps)

    state = WorkspaceState(repo_root=repo, request=request)
    orchestrator = Orchestrator(
        state,
        agent_config=config,
        config=OrchestratorConfig(
            max_steps=config.max_steps,
            max_repair_attempts=config.max_repair_attempts,
            dry_run=bool(args.dry_run),
        ),
    )
    result_state = orchestrator.run()
    report = result_state.to_report_dict()
    print(json.dumps(report, indent=2))

    if args.report_json:
        report_path = Path(args.report_json)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logging.getLogger(__name__).info("Wrote report to %s", report_path)

    if result_state.errors and (result_state.final_report is None or not result_state.final_report.success):
        return 1
    if result_state.final_report is not None and not result_state.final_report.success:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
