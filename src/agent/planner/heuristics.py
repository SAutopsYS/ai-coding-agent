"""Deterministic heuristics for intent detection and file relevance scoring.

No LLM calls. Keyword and path-pattern rules only, kept generic across stacks.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

from agent.explorer.models import RepositorySummary
from agent.planner.models import RelevantFile

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IntentRule:
    """Maps request keywords to a planning intent."""

    intent: str
    keywords: tuple[str, ...]
    preferred_roles: tuple[str, ...]
    description: str


INTENT_RULES: tuple[IntentRule, ...] = (
    IntentRule(
        intent="search",
        keywords=(
            "search",
            "find",
            "query",
            "filter",
            "lookup",
            "full-text",
            "fulltext",
        ),
        preferred_roles=("controller", "route", "service", "model"),
        description="Request involves searching or filtering data.",
    ),
    IntentRule(
        intent="organize",
        keywords=(
            "tag",
            "tags",
            "label",
            "labels",
            "category",
            "categories",
            "folder",
            "folders",
            "organize",
            "organise",
            "organization",
            "organisation",
        ),
        preferred_roles=("model", "controller", "route", "service"),
        description="Request involves organizing or categorizing entities.",
    ),
    IntentRule(
        intent="authentication",
        keywords=(
            "auth",
            "authentication",
            "authorization",
            "login",
            "logout",
            "signup",
            "sign-up",
            "register",
            "jwt",
            "session",
            "oauth",
            "password",
            "middleware",
        ),
        preferred_roles=("middleware", "route", "controller", "config", "model"),
        description="Request involves authentication or access control.",
    ),
    IntentRule(
        intent="crud",
        keywords=(
            "create",
            "update",
            "delete",
            "remove",
            "edit",
            "crud",
            "endpoint",
            "api",
        ),
        preferred_roles=("controller", "route", "model", "service"),
        description="Request involves create/read/update/delete API behavior.",
    ),
    IntentRule(
        intent="persistence",
        keywords=(
            "schema",
            "model",
            "database",
            "migration",
            "mongo",
            "sql",
            "field",
            "attribute",
        ),
        preferred_roles=("model", "config", "migration"),
        description="Request involves data shape or persistence changes.",
    ),
    IntentRule(
        intent="documentation",
        keywords=("readme", "document", "docs", "documentation"),
        preferred_roles=("readme", "docs"),
        description="Request involves documentation updates.",
    ),
    IntentRule(
        intent="containerization",
        keywords=("docker", "container", "compose", "deploy"),
        preferred_roles=("container", "config"),
        description="Request involves containers or deployment packaging.",
    ),
    IntentRule(
        intent="testing",
        keywords=("test", "tests", "unit test", "integration test", "coverage"),
        preferred_roles=("test", "controller", "service"),
        description="Request involves tests.",
    ),
)


ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("model", re.compile(r"(^|/)(models?|entities|schemas?)(/|$)", re.I)),
    ("model", re.compile(r"\.(model|schema|entity)\.[^.]+$", re.I)),
    ("controller", re.compile(r"(^|/)(controllers?|handlers?)(/|$)", re.I)),
    ("controller", re.compile(r"\.(controller|handler)\.[^.]+$", re.I)),
    ("route", re.compile(r"(^|/)(routes?|routers?|endpoints?)(/|$)", re.I)),
    ("route", re.compile(r"\.(routes?|router)\.[^.]+$", re.I)),
    ("service", re.compile(r"(^|/)(services?|usecases?|use_cases?)(/|$)", re.I)),
    ("service", re.compile(r"\.(service)\.[^.]+$", re.I)),
    ("middleware", re.compile(r"(^|/)(middleware|middlewares|guards?|auth)(/|$)", re.I)),
    ("middleware", re.compile(r"(auth|middleware|guard)", re.I)),
    ("migration", re.compile(r"(^|/)(migrations?|alembic|flyway)(/|$)", re.I)),
    ("config", re.compile(r"(^|/)(config|configs|settings)(/|$)", re.I)),
    ("config", re.compile(r"(config|settings)\.[^.]+$", re.I)),
    ("test", re.compile(r"(^|/)(tests?|spec|__tests__)(/|$)", re.I)),
    ("test", re.compile(r"(\.test\.|\.spec\.|test_)", re.I)),
    ("readme", re.compile(r"(^|/)readme(\.|$)", re.I)),
    ("container", re.compile(r"(^|/)dockerfile$|(^|/)docker-compose\.ya?ml$", re.I)),
    ("manifest", re.compile(
        r"(^|/)(package\.json|pyproject\.toml|requirements\.txt|go\.mod|cargo\.toml|pom\.xml)$",
        re.I,
    )),
    ("entrypoint", re.compile(
        r"(^|/)(server|main|app|index|manage|wsgi|asgi)\.[^.]+$",
        re.I,
    )),
)


def normalize_request(request: str) -> str:
    """Lowercase and collapse whitespace for keyword matching."""
    return re.sub(r"\s+", " ", request.strip().lower())


def detect_intents(request: str) -> list[str]:
    """Return ordered unique intents inferred from the request text."""
    text = normalize_request(request)
    found: list[str] = []
    for rule in INTENT_RULES:
        if any(re.search(rf"\b{re.escape(keyword)}\b", text) for keyword in rule.keywords):
            if rule.intent not in found:
                found.append(rule.intent)
                logger.info("Detected intent '%s' from request keywords", rule.intent)

    if "organize" in found and "persistence" not in found:
        found.append("persistence")
        logger.info("Added derived intent 'persistence' because organize was detected")

    if not found:
        found.append("crud")
        logger.info("No specific intent matched; defaulting to 'crud'")
    return found


def infer_role(path: str) -> str:
    """Infer a coarse file role from its relative path."""
    posix = path.replace("\\", "/")
    for role, pattern in ROLE_PATTERNS:
        if pattern.search(posix):
            return role
    return "unknown"


def _candidate_paths(summary: RepositorySummary) -> list[str]:
    """Collect candidate file paths from the summary map and important files."""
    paths: set[str] = set()
    if summary.repository_map is not None:
        paths.update(summary.repository_map.files)
    for important in summary.important_files:
        paths.add(important.path)
    if summary.entry_point:
        paths.add(summary.entry_point)
    return sorted(paths)


def _role_boost(role: str, intents: list[str]) -> int:
    boost = 0
    for intent in intents:
        for rule in INTENT_RULES:
            if rule.intent == intent and role in rule.preferred_roles:
                index = rule.preferred_roles.index(role)
                boost += max(1, 4 - index)
    return boost


def _path_keyword_boost(path: str, request: str) -> int:
    """Boost files whose names overlap request tokens (e.g. note, user)."""
    text = normalize_request(request)
    tokens = [t for t in re.findall(r"[a-z0-9_]+", text) if len(t) >= 4]
    stop = {
        "with",
        "that",
        "this",
        "from",
        "have",
        "user",
        "users",
        "make",
        "able",
        "their",
        "better",
        "application",
        "improve",
    }
    tokens = [t for t in tokens if t not in stop]
    path_l = path.lower()
    score = 0
    for token in tokens:
        if token in path_l:
            score += 2
    return score


def select_relevant_files(
    summary: RepositorySummary,
    request: str,
    intents: list[str],
    *,
    max_files: int = 20,
) -> list[RelevantFile]:
    """Score and select relevant files with explicit reasons."""
    preferred_roles: set[str] = set()
    for intent in intents:
        for rule in INTENT_RULES:
            if rule.intent == intent:
                preferred_roles.update(rule.preferred_roles)

    context_roles = {"entrypoint", "manifest", "readme", "config"}

    scored: list[RelevantFile] = []
    for path in _candidate_paths(summary):
        role = infer_role(path)
        for important in summary.important_files:
            if important.path == path and role == "unknown":
                role = important.role.split(",")[0].strip() or role

        score = 0
        reasons: list[str] = []

        role_score = _role_boost(role, intents)
        if role_score:
            score += role_score
            reasons.append(
                f"Path role '{role}' matches intent(s) {', '.join(intents)} "
                f"(preferred for this change)."
            )

        keyword_score = _path_keyword_boost(path, request)
        if keyword_score:
            score += keyword_score
            reasons.append("Filename/path overlaps meaningful tokens from the product request.")

        if role in context_roles:
            score += 1
            reasons.append(
                f"Included as project context ({role}) so the plan respects entry/config/docs."
            )

        parent = str(Path(path).parent).replace("\\", "/")
        if parent in summary.important_directories or any(
            parent == d or parent.startswith(f"{d}/") for d in summary.important_directories
        ):
            if role in preferred_roles:
                score += 1
                reasons.append("File lives under an important application directory.")

        if score <= 0:
            continue

        unique_reasons: list[str] = []
        for reason in reasons:
            if reason not in unique_reasons:
                unique_reasons.append(reason)

        scored.append(
            RelevantFile(
                path=path,
                reason=" ".join(unique_reasons),
                priority=score,
                role_hint=role,
            )
        )

    scored.sort(key=lambda item: (-item.priority, item.path))
    selected = scored[:max_files]
    logger.info("Selected %d relevant files (from %d scored)", len(selected), len(scored))
    for item in selected:
        logger.debug("Relevant file %s (priority=%d): %s", item.path, item.priority, item.reason)
    return selected


def build_goal(request: str, summary: RepositorySummary, intents: list[str]) -> str:
    """Compose a concise goal statement."""
    intent_text = ", ".join(intents)
    framework = summary.framework or summary.project_type
    return (
        f"Satisfy the product request for '{summary.project_name}' "
        f"({framework}): {request.strip()} "
        f"[detected intents: {intent_text}]"
    )


def build_assumptions(summary: RepositorySummary, intents: list[str]) -> list[str]:
    """List planning assumptions derived from the repository summary."""
    assumptions = [
        f"Primary project type is {summary.project_type} "
        f"(language: {summary.detected_language}).",
        "Changes should extend existing architecture rather than rewrite the application.",
        "No authentication/multi-tenancy exists unless the request explicitly requires it "
        "or auth files are already present.",
        "Planning is heuristic-only; exact APIs/fields will be confirmed during implementation.",
    ]
    if summary.framework:
        assumptions.append(f"Detected framework '{summary.framework}' should be preserved.")
    if summary.entry_point:
        assumptions.append(f"Application entry point is '{summary.entry_point}'.")
    if "search" in intents:
        assumptions.append(
            "Search can be implemented via query parameters or a dedicated search endpoint "
            "consistent with existing routing style."
        )
    if "organize" in intents:
        assumptions.append(
            "Organization features likely require schema fields (e.g. tags/categories) "
            "plus API support to set and filter them."
        )
    if "authentication" in intents:
        assumptions.append(
            "Authentication may require new middleware and protected routes; "
            "existing public endpoints must be reviewed carefully."
        )
    if summary.repository_map and summary.repository_map.file_count == 0:
        assumptions.append("Repository map is empty; plan confidence is low.")
    return assumptions
