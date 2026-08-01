"""Post-edit verification: syntax, feature presence, preserved APIs."""

from __future__ import annotations

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from agent.explorer.models import RepositorySummary
from agent.planner.models import ExecutionPlan
from agent.state import FileChange
from agent.tools.base import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class VerificationResult:
    """Outcome of verification checks."""

    syntax_ok: bool = True
    feature_implemented: bool = False
    existing_preserved: bool = True
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """True when syntax, feature, and preservation checks all pass."""
        return self.syntax_ok and self.feature_implemented and self.existing_preserved

    def as_notes(self) -> str:
        """Format pass/fail checks as a multi-line note string."""
        lines = ["Verification results:"]
        lines.extend(f"- PASS: {c}" for c in self.checks)
        lines.extend(f"- FAIL: {f}" for f in self.failures)
        return "\n".join(lines)


class Verifier:
    """Heuristic verifier that does not require an LLM."""

    def __init__(self, repo_root: Path, tools: ToolRegistry | None = None) -> None:
        self.repo_root = repo_root.resolve()
        self.tools = tools

    def verify(
        self,
        *,
        summary: RepositorySummary,
        plan: ExecutionPlan,
        files_changed: list[FileChange],
        request: str,
    ) -> VerificationResult:
        """Run syntax, feature, and preservation checks after edits."""
        logger.info("Running verification on %s", self.repo_root)
        result = VerificationResult()

        syntax_ok, syntax_notes = self._check_syntax(files_changed)
        result.syntax_ok = syntax_ok
        if syntax_ok:
            result.checks.append("No syntax errors detected in modified source files.")
        else:
            result.failures.extend(syntax_notes)

        preserved, preserve_notes = self._check_preserved_functionality(summary, plan)
        result.existing_preserved = preserved
        if preserved:
            result.checks.append("Existing core entrypoints/routes/handlers still present.")
        else:
            result.failures.extend(preserve_notes)

        implemented, feature_notes = self._check_feature_implemented(
            plan=plan,
            request=request,
            files_changed=files_changed,
        )
        result.feature_implemented = implemented
        if implemented:
            result.checks.append("Requested feature signals found in the repository.")
        else:
            result.failures.extend(feature_notes)

        if self.tools is not None:
            tool_names = {t["name"] for t in self.tools.list_tools()}
            if "git_diff" in tool_names:
                diff = self.tools.run("git_diff", {})
                if diff.success:
                    result.checks.append("Captured git diff for workspace changes.")

        logger.info(
            "Verification finished success=%s syntax=%s feature=%s preserved=%s",
            result.success,
            result.syntax_ok,
            result.feature_implemented,
            result.existing_preserved,
        )
        return result

    def _check_syntax(self, files_changed: list[FileChange]) -> tuple[bool, list[str]]:
        failures: list[str] = []
        for change in files_changed:
            path = self.repo_root / change.path
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            try:
                text = path.read_text(encoding="utf-8")
            except OSError as exc:
                failures.append(f"Cannot read {change.path}: {exc}")
                continue

            if suffix == ".py":
                try:
                    ast.parse(text)
                except SyntaxError as exc:
                    failures.append(f"Python syntax error in {change.path}: {exc}")
            elif suffix in {".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"}:
                if not self._balanced_js_brackets(text):
                    failures.append(f"Unbalanced brackets in {change.path}")
        return len(failures) == 0, failures

    def _balanced_js_brackets(self, text: str) -> bool:
        pairs = {"(": ")", "[": "]", "{": "}"}
        stack: list[str] = []
        in_single = False
        in_double = False
        in_line_comment = False
        in_block_comment = False
        i = 0
        while i < len(text):
            ch = text[i]
            nxt = text[i + 1] if i + 1 < len(text) else ""
            if in_line_comment:
                if ch == "\n":
                    in_line_comment = False
                i += 1
                continue
            if in_block_comment:
                if ch == "*" and nxt == "/":
                    in_block_comment = False
                    i += 2
                    continue
                i += 1
                continue
            if not in_single and not in_double and ch == "/" and nxt == "/":
                in_line_comment = True
                i += 2
                continue
            if not in_single and not in_double and ch == "/" and nxt == "*":
                in_block_comment = True
                i += 2
                continue
            if ch == "'" and not in_double:
                in_single = not in_single
                i += 1
                continue
            if ch == '"' and not in_single:
                in_double = not in_double
                i += 1
                continue
            if in_single or in_double:
                i += 1
                continue
            if ch in pairs:
                stack.append(pairs[ch])
            elif ch in pairs.values():
                if not stack or stack.pop() != ch:
                    return False
            i += 1
        return not stack

    def _check_preserved_functionality(
        self,
        summary: RepositorySummary,
        plan: ExecutionPlan,
    ) -> tuple[bool, list[str]]:
        failures: list[str] = []
        if summary.entry_point:
            entry = self.repo_root / summary.entry_point
            if not entry.is_file():
                failures.append(f"Entry point missing after edits: {summary.entry_point}")

        for item in plan.relevant_files:
            if item.role_hint not in {"route", "controller", "model", "entrypoint"}:
                continue
            absolute = self.repo_root / item.path
            if not absolute.is_file():
                failures.append(f"Relevant file missing after edits: {item.path}")
                continue

            text = absolute.read_text(encoding="utf-8", errors="ignore")
            if item.role_hint == "controller" and "exports." in text:
                crud = [
                    name
                    for name in ("create", "findAll", "findOne", "update", "delete")
                    if f"exports.{name}" in text
                ]
                if len(crud) < 4:
                    failures.append(
                        f"Core CRUD exports appear incomplete in {item.path}: {crud}"
                    )
            if item.role_hint == "route" and re.search(
                r"app\.(get|post|put|delete)\(", text
            ):
                present = [
                    method
                    for method in ("get", "post", "put", "delete")
                    if f"app.{method}" in text
                ]
                if len(present) < 3:
                    failures.append(
                        f"Expected multiple HTTP verbs preserved in {item.path}"
                    )

        return len(failures) == 0, failures

    def _check_feature_implemented(
        self,
        *,
        plan: ExecutionPlan,
        request: str,
        files_changed: list[FileChange],
    ) -> tuple[bool, list[str]]:
        if not files_changed and plan.detected_intents:
            return False, ["No files were modified although the plan expected changes."]

        text_blobs: list[str] = []
        scanned: set[str] = set()
        for change in files_changed:
            path = self.repo_root / change.path
            if path.is_file():
                text_blobs.append(path.read_text(encoding="utf-8", errors="ignore"))
                scanned.add(change.path)
        for item in plan.relevant_files:
            if item.path in scanned:
                continue
            path = self.repo_root / item.path
            if path.is_file():
                text_blobs.append(path.read_text(encoding="utf-8", errors="ignore"))

        corpus = "\n".join(text_blobs).lower()
        request_l = request.lower()
        intents = set(plan.detected_intents)
        failures: list[str] = []
        ok = True

        needs_organize = "organize" in intents or any(
            k in request_l for k in ("tag", "organise", "organize")
        )
        needs_search = "search" in intents or "search" in request_l

        if needs_organize and "tags" not in corpus:
            ok = False
            failures.append("Organization intent detected but no 'tags' field/usage found.")
        if needs_search and not any(
            token in corpus
            for token in ("search", "regex", "$regex", "req.query", "query.q", "filter")
        ):
            ok = False
            failures.append(
                "Search intent detected but no search/filter implementation markers found."
            )

        if not needs_organize and not needs_search and not files_changed:
            ok = False
            failures.append("No actionable feature evidence found.")

        return ok, failures
