"""Deterministic execution planner.

Builds an ExecutionPlan from a RepositorySummary and a product request.
Does not edit files, run shell commands, or call an LLM.
"""

from __future__ import annotations

import logging
from collections import defaultdict

from agent.explorer.models import RepositorySummary
from agent.planner.heuristics import (
    build_assumptions,
    build_goal,
    detect_intents,
    select_relevant_files,
)
from agent.planner.models import (
    ExecutionPlan,
    PlanStep,
    RelevantFile,
    Risk,
    ValidationItem,
)

logger = logging.getLogger(__name__)


class ExecutionPlanner:
    """Create a structured execution plan using repository heuristics only."""

    def __init__(self, *, max_relevant_files: int = 20) -> None:
        """Initialize planner settings.

        Args:
            max_relevant_files: Cap on relevant files included in the plan.
        """
        self.max_relevant_files = max_relevant_files

    def create_plan(self, summary: RepositorySummary, request: str) -> ExecutionPlan:
        """Generate an ExecutionPlan for the given summary and request.

        Args:
            summary: Output of RepositoryExplorer (or equivalent).
            request: Natural-language product request.

        Returns:
            A fully populated ExecutionPlan.

        Raises:
            ValueError: If the request is empty/blank.
        """
        cleaned = request.strip()
        if not cleaned:
            raise ValueError("User request must be a non-empty string.")

        logger.info(
            "Creating execution plan for project=%s request=%r",
            summary.project_name,
            cleaned,
        )

        intents = detect_intents(cleaned)
        relevant_files = select_relevant_files(
            summary,
            cleaned,
            intents,
            max_files=self.max_relevant_files,
        )
        steps = self._build_steps(summary, cleaned, intents, relevant_files)
        risks = self._build_risks(summary, intents, relevant_files)
        validation = self._build_validation_checklist(summary, intents)
        unchanged = self._build_unchanged_functionality(summary, intents)

        plan = ExecutionPlan(
            goal=build_goal(cleaned, summary, intents),
            assumptions=build_assumptions(summary, intents),
            relevant_files=relevant_files,
            steps=steps,
            risks=risks,
            validation_checklist=validation,
            unchanged_functionality=unchanged,
            detected_intents=intents,
            project_type=summary.project_type,
            request=cleaned,
        )

        logger.info(
            "Plan created: intents=%s steps=%d relevant_files=%d risks=%d",
            intents,
            len(plan.steps),
            len(plan.relevant_files),
            len(plan.risks),
        )
        return plan

    def _files_by_role(self, relevant_files: list[RelevantFile]) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = defaultdict(list)
        for item in relevant_files:
            grouped[item.role_hint].append(item.path)
        return grouped

    def _build_steps(
        self,
        summary: RepositorySummary,
        request: str,
        intents: list[str],
        relevant_files: list[RelevantFile],
    ) -> list[PlanStep]:
        by_role = self._files_by_role(relevant_files)
        steps: list[PlanStep] = []
        step_no = 1

        def add(
            title: str,
            description: str,
            files: list[str],
            rationale: str,
        ) -> None:
            nonlocal step_no
            steps.append(
                PlanStep(
                    id=f"step-{step_no}",
                    title=title,
                    description=description,
                    files=files,
                    rationale=rationale,
                )
            )
            step_no += 1

        context_files = self._dedupe_paths(
            [
                *by_role.get("entrypoint", []),
                *by_role.get("manifest", []),
                *by_role.get("readme", []),
            ]
        )
        add(
            title="Review current architecture touchpoints",
            description=(
                "Confirm entrypoint, manifests, and README against the product request "
                "before changing behavior."
            ),
            files=context_files,
            rationale="Ground the change in the existing project layout and stack.",
        )

        if "organize" in intents or "persistence" in intents:
            add(
                title="Extend the data model for organization fields",
                description=(
                    "Add or adjust persistence fields needed for organization "
                    "(for example tags, categories, or labels) while keeping existing "
                    "required fields valid."
                ),
                files=by_role.get("model", []),
                rationale=(
                    "Organization features usually require schema-level changes; "
                    "model/entity files are the safest first implementation surface."
                ),
            )

        if "search" in intents:
            add(
                title="Implement search/filter behavior in application logic",
                description=(
                    "Add query handling that can search or filter records by the fields "
                    "implied by the request (title/content/tags/etc.), reusing existing "
                    "controller/service patterns."
                ),
                files=[
                    *by_role.get("controller", []),
                    *by_role.get("service", []),
                    *by_role.get("model", []),
                ],
                rationale=(
                    "Search requests prioritize controllers/services (and models for query "
                    "construction) rather than unrelated UI or infra files."
                ),
            )

        if "authentication" in intents:
            add(
                title="Add or update authentication middleware and protected routes",
                description=(
                    "Introduce or extend auth middleware and wire it into routes that "
                    "must require identity, without breaking intentionally public endpoints."
                ),
                files=[
                    *by_role.get("middleware", []),
                    *by_role.get("route", []),
                    *by_role.get("controller", []),
                    *by_role.get("config", []),
                ],
                rationale=(
                    "Authentication changes concentrate on middleware, routes, and config."
                ),
            )

        if "crud" in intents and "search" not in intents and "organize" not in intents:
            add(
                title="Update CRUD handlers for the requested behavior",
                description=(
                    "Modify create/read/update/delete handlers and supporting models "
                    "to implement the requested API behavior."
                ),
                files=[
                    *by_role.get("controller", []),
                    *by_role.get("route", []),
                    *by_role.get("model", []),
                    *by_role.get("service", []),
                ],
                rationale="Generic API changes map to controller/route/model layers.",
            )

        if any(i in intents for i in ("search", "organize", "authentication", "crud")):
            route_files = by_role.get("route", [])
            if route_files or summary.project_type in {"Node.js", "Python", "Java", "Go"}:
                add(
                    title="Wire HTTP routes/API surface",
                    description=(
                        "Expose the new or updated behavior through routes/endpoints, "
                        "preserving existing route paths and response shapes where possible. "
                        "If adding a path like '/resource/search', register it before "
                        "parameterized '/resource/:id' routes."
                    ),
                    files=route_files or by_role.get("controller", []),
                    rationale=(
                        "Routes are the public contract; search/organize/auth features "
                        "must be reachable without breaking current clients."
                    ),
                )

        if "testing" in intents or by_role.get("test"):
            add(
                title="Add or update automated tests",
                description=(
                    "Cover the new behavior and assert existing critical flows still pass."
                ),
                files=by_role.get("test", []),
                rationale="Tests reduce regression risk for API and data-model changes.",
            )

        if "documentation" in intents or by_role.get("readme"):
            add(
                title="Update documentation for the new behavior",
                description=(
                    "Document new fields/endpoints/query params in README or API docs."
                ),
                files=by_role.get("readme", []) + by_role.get("docs", []),
                rationale="Docs keep the public API discoverable after the change.",
            )

        if "containerization" in intents:
            add(
                title="Update container/deployment configuration if required",
                description=(
                    "Adjust Dockerfile or compose settings only if the feature needs "
                    "runtime/env changes."
                ),
                files=by_role.get("container", []) + by_role.get("config", []),
                rationale="Container files matter when deployment or env wiring changes.",
            )

        add(
            title="Regression-check existing functionality",
            description=(
                "Manually or automatically verify that previously supported operations "
                "still succeed with unchanged request/response contracts."
            ),
            files=[
                *by_role.get("route", []),
                *by_role.get("controller", []),
                *by_role.get("entrypoint", []),
            ],
            rationale=(
                f"Request is additive for '{summary.project_name}'; "
                "existing clients must keep working."
            ),
        )

        for step in steps:
            step.files = self._dedupe_paths(step.files)

        logger.info("Built %d plan steps for request intents=%s", len(steps), intents)
        return steps

    @staticmethod
    def _dedupe_paths(paths: list[str]) -> list[str]:
        """Return paths deduplicated while preserving order."""
        deduped: list[str] = []
        seen: set[str] = set()
        for path in paths:
            if path not in seen:
                deduped.append(path)
                seen.add(path)
        return deduped

    def _build_risks(
        self,
        summary: RepositorySummary,
        intents: list[str],
        relevant_files: list[RelevantFile],
    ) -> list[Risk]:
        risks: list[Risk] = [
            Risk(
                description=(
                    "Heuristic planning may miss relevant files not matching common "
                    "naming patterns."
                ),
                severity="medium",
                mitigation=(
                    "During implementation, re-scan for handlers/models related to the "
                    "target domain entity."
                ),
            )
        ]

        if "search" in intents:
            risks.append(
                Risk(
                    description=(
                        "A new '/search' route can be captured by a parameterized "
                        "'/:id' route if registration order is wrong."
                    ),
                    severity="high",
                    mitigation=(
                        "Register static search routes before parameterized identity routes."
                    ),
                )
            )
            risks.append(
                Risk(
                    description=(
                        "Unindexed search over large collections may become slow."
                    ),
                    severity="medium",
                    mitigation=(
                        "Start with simple filters; add DB indexes if the dataset grows."
                    ),
                )
            )

        if "organize" in intents:
            risks.append(
                Risk(
                    description=(
                        "Schema additions can break clients that reject unknown fields "
                        "or assume a fixed document shape."
                    ),
                    severity="medium",
                    mitigation=(
                        "Keep new fields optional with safe defaults; avoid renaming "
                        "existing fields."
                    ),
                )
            )

        if "authentication" in intents:
            risks.append(
                Risk(
                    description=(
                        "Auth middleware can accidentally lock down previously public APIs."
                    ),
                    severity="high",
                    mitigation=(
                        "Apply auth selectively; explicitly preserve public health/docs "
                        "and any currently open CRUD routes unless the request says otherwise."
                    ),
                )
            )

        roles = {f.role_hint for f in relevant_files}
        if "model" not in roles and ("organize" in intents or "persistence" in intents):
            risks.append(
                Risk(
                    description="No model/schema files were identified for persistence changes.",
                    severity="high",
                    mitigation=(
                        "Locate entity definitions manually before editing "
                        f"(project type: {summary.project_type})."
                    ),
                )
            )

        if not relevant_files:
            risks.append(
                Risk(
                    description="No relevant files were selected; plan confidence is low.",
                    severity="high",
                    mitigation="Re-run exploration or broaden path role heuristics.",
                )
            )

        return risks

    def _build_validation_checklist(
        self,
        summary: RepositorySummary,
        intents: list[str],
    ) -> list[ValidationItem]:
        items: list[ValidationItem] = [
            ValidationItem(
                description=(
                    "Existing create/read/update/delete flows still succeed with the "
                    "same primary request fields."
                ),
                category="regression",
            ),
            ValidationItem(
                description=(
                    f"Application still starts from entry point "
                    f"'{summary.entry_point or 'unknown'}'."
                ),
                category="runtime",
            ),
        ]

        if "search" in intents:
            items.append(
                ValidationItem(
                    description=(
                        "Search/filter returns matching records and handles empty results cleanly."
                    ),
                    category="api",
                )
            )
            items.append(
                ValidationItem(
                    description="Search does not break get-by-id for valid identifiers.",
                    category="regression",
                )
            )

        if "organize" in intents:
            items.append(
                ValidationItem(
                    description=(
                        "Organization fields can be set on create/update and read back "
                        "on fetch/list."
                    ),
                    category="data",
                )
            )
            items.append(
                ValidationItem(
                    description=(
                        "Records without organization fields still load (backward compatible)."
                    ),
                    category="regression",
                )
            )

        if "authentication" in intents:
            items.append(
                ValidationItem(
                    description="Protected routes reject unauthenticated requests.",
                    category="api",
                )
            )
            items.append(
                ValidationItem(
                    description="Authenticated happy-path requests still succeed.",
                    category="api",
                )
            )

        items.append(
            ValidationItem(
                description="README or API notes mention new parameters/fields if user-facing.",
                category="docs",
            )
        )
        return items

    def _build_unchanged_functionality(
        self,
        summary: RepositorySummary,
        intents: list[str],
    ) -> list[str]:
        unchanged = [
            "Existing HTTP methods and paths that are not explicitly being replaced.",
            "Current required validation rules for core create/update payloads.",
            "Existing persistence of current primary fields (do not rename/remove them).",
            f"Detected stack components remain in use: "
            f"{', '.join(summary.technology_stack) if summary.technology_stack else summary.project_type}.",
        ]
        if summary.framework:
            unchanged.append(f"{summary.framework} remains the primary application framework.")
        if "authentication" not in intents:
            unchanged.append(
                "Access control posture remains unchanged (no unexpected auth requirements)."
            )
        if "search" in intents or "organize" in intents:
            unchanged.append(
                "Basic list/get-by-id behavior remains available for existing clients."
            )
        return unchanged
