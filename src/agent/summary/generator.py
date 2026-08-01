"""Generate the final agent report."""

from __future__ import annotations

import logging

from agent.llm.schemas import FinalReport
from agent.planner.models import ExecutionPlan
from agent.prompts.summarize import build_heuristic_summary
from agent.state import FileChange
from agent.verify.verifier import VerificationResult

logger = logging.getLogger(__name__)


class SummaryGenerator:
    """Build a FinalReport from plan, edits, and verification results."""

    def generate(
        self,
        *,
        request: str,
        plan: ExecutionPlan,
        files_changed: list[FileChange],
        verification: VerificationResult,
    ) -> FinalReport:
        """Build the final report from plan, edits, and verification results."""
        logger.info("Generating final report for %d changed files", len(files_changed))
        heuristic = build_heuristic_summary(
            request=request,
            files_changed=files_changed,
            validation_notes=verification.failures + verification.checks,
            preserved=plan.unchanged_functionality,
        )

        files_modified = [
            {"path": change.path, "why": change.summary or change.change_type}
            for change in files_changed
        ]
        validation_results = [
            *[{"status": "pass", "detail": item} for item in verification.checks],
            *[{"status": "fail", "detail": item} for item in verification.failures],
        ]

        report = FinalReport(
            overview=str(heuristic["overview"]),
            files_modified=files_modified,
            summary_of_changes=list(heuristic["behavior_added"]),  # type: ignore[arg-type]
            validation_results=validation_results,
            preserved_functionality=plan.unchanged_functionality,
            follow_ups=[
                risk.description for risk in plan.risks if risk.severity in {"medium", "high"}
            ][:5],
            success=verification.success,
        )
        return report
