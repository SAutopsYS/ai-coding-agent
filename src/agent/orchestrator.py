"""Stage machine that drives the full agent pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from agent.config_loader import AgentConfig
from agent.explorer import RepositoryExplorer, RepositorySummary
from agent.llm.client import ChatMessage, LLMClient, LLMClientConfig
from agent.llm.parse import parse_model
from agent.llm.schemas import EditInstructions, FileEdit
from agent.planner import ExecutionPlan, ExecutionPlanner
from agent.prompts.act import build_act_messages
from agent.safety.limits import LimitExceeded, Limits
from agent.state import PlanStep, WorkspaceState
from agent.summary import SummaryGenerator
from agent.tools.base import ToolRegistry
from agent.tools.factory import build_tool_registry
from agent.verify import Verifier

logger = logging.getLogger(__name__)


@dataclass
class OrchestratorConfig:
    """Runtime knobs for a single orchestrator run."""

    max_steps: int = 30
    max_repair_attempts: int = 1
    dry_run: bool = False


class Orchestrator:
    """Coordinates explore, plan, LLM edits, verify, and summarize."""

    def __init__(
        self,
        state: WorkspaceState,
        *,
        agent_config: AgentConfig,
        llm_client: LLMClient | None = None,
        tools: ToolRegistry | None = None,
        config: OrchestratorConfig | None = None,
    ) -> None:
        self.state = state
        self.agent_config = agent_config
        self.config = config or OrchestratorConfig(
            max_steps=agent_config.max_steps,
            max_repair_attempts=agent_config.max_repair_attempts,
        )
        self.limits = Limits(
            max_steps=agent_config.max_steps,
            max_files_touched=agent_config.max_files_touched,
            max_read_bytes=agent_config.max_read_bytes,
            max_write_bytes=agent_config.max_write_bytes,
            max_repair_attempts=agent_config.max_repair_attempts,
        )
        self.llm = llm_client or LLMClient(
            LLMClientConfig(
                provider=agent_config.llm_provider,
                model=agent_config.llm_model,
                temperature=agent_config.llm_temperature,
                api_key=agent_config.llm_api_key,
                base_url=agent_config.llm_base_url,
                extra={"max_retries": agent_config.llm_max_retries},
            )
        )
        self.tools = tools or build_tool_registry(
            state.repo_root,
            agent_config,
            dry_run=self.config.dry_run,
        )
        self._summary: RepositorySummary | None = None
        self._plan: ExecutionPlan | None = None

    def run(self) -> WorkspaceState:
        """Execute the full agent pipeline and return the updated state."""
        logger.info("Orchestrator starting for %s", self.state.repo_root)
        try:
            self.stage_explore()
            self.stage_plan()
            self.stage_select_relevant_files()
            self.stage_act()
            self.stage_verify()
            self.stage_summarize()
        except Exception as exc:  # noqa: BLE001
            logger.exception("Orchestrator failed")
            self.state.errors.append(str(exc))
            if self.state.final_summary == "":
                self.state.final_summary = f"Agent failed: {exc}"
        logger.info("Orchestrator finished with %d file changes", len(self.state.files_changed))
        return self.state

    def stage_explore(self) -> None:
        """Explore the repository with the heuristic explorer."""
        logger.info("Stage: explore")
        explorer = RepositoryExplorer(self.state.repo_root)
        summary = explorer.explore()
        self._summary = summary
        self.state.repository_summary = summary
        self.state.file_tree_summary = summary.repository_tree_summary
        self.state.stack_summary = ", ".join(summary.technology_stack)
        self.state.entrypoint = summary.entry_point or ""

    def stage_plan(self) -> None:
        """Create a deterministic execution plan."""
        logger.info("Stage: plan")
        if self._summary is None:
            raise RuntimeError("Explore stage must run before plan")
        planner = ExecutionPlanner()
        plan = planner.create_plan(self._summary, self.state.request)
        self._plan = plan
        self.state.execution_plan = plan
        self.state.plan_steps = [
            PlanStep(id=step.id, description=step.title, files=list(step.files))
            for step in plan.steps
        ]
        self.state.invariants = list(plan.unchanged_functionality)
        self.state.risks = [risk.description for risk in plan.risks]

    def stage_select_relevant_files(self) -> None:
        """Load contents for planned relevant files via filesystem tools."""
        logger.info("Stage: select relevant files")
        if self._plan is None:
            raise RuntimeError("Plan stage must run before selecting files")

        paths = [item.path for item in self._plan.relevant_files]
        self.state.relevant_files = paths
        excerpts: dict[str, str] = {}
        for path in paths:
            result = self.tools.run("read_file", {"path": path})
            self.state.append_tool_result(
                "read_file",
                {"path": path},
                result.output if result.success else (result.error or ""),
                success=result.success,
            )
            if result.success:
                excerpts[path] = result.output
        self.state.file_excerpts = excerpts
        logger.info("Loaded %d relevant file excerpts", len(excerpts))

    def stage_act(self) -> None:
        """Call LLM for structured edits and apply them through tools."""
        logger.info("Stage: act")
        if self._summary is None or self._plan is None:
            raise RuntimeError("Explore/plan required before act")

        messages = build_act_messages(
            request=self.state.request,
            summary=self._summary,
            plan=self._plan,
            file_contents=self.state.file_excerpts,
            available_tools=[item["name"] for item in self.tools.list_tools()],
        )

        repair_attempts = 0
        while True:
            self.state.step_count += 1
            try:
                self.limits.check_step_count(self.state.step_count)
            except LimitExceeded as exc:
                self.state.errors.append(str(exc))
                break

            raw = self.llm.complete(messages)
            try:
                instructions = parse_model(raw, EditInstructions)
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to parse EditInstructions: %s", exc)
                self.state.errors.append(f"Invalid LLM edit JSON: {exc}")
                if repair_attempts >= self.config.max_repair_attempts:
                    break
                repair_attempts += 1
                messages = [
                    *messages,
                    ChatMessage(role="assistant", content=raw),
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous response was invalid. "
                            f"Error: {exc}. Return ONLY valid EditInstructions JSON."
                        ),
                    ),
                ]
                continue

            applied_any = self._apply_edits(instructions.edits)
            failed_edits = any("failed for" in err for err in self.state.errors)
            if instructions.done and not failed_edits:
                break
            if not applied_any and not failed_edits:
                break
            if repair_attempts >= self.config.max_repair_attempts:
                break
            repair_attempts += 1
            self.stage_select_relevant_files()
            messages = build_act_messages(
                request=self.state.request,
                summary=self._summary,
                plan=self._plan,
                file_contents=self.state.file_excerpts,
                available_tools=[item["name"] for item in self.tools.list_tools()],
            )
            messages.extend(
                [
                    ChatMessage(role="assistant", content=raw),
                    ChatMessage(
                        role="user",
                        content=(
                            "Some edits failed or more work may remain. "
                            "Return additional EditInstructions JSON or done=true with edits=[]."
                        ),
                    ),
                ]
            )

    def _apply_edits(self, edits: list[FileEdit]) -> bool:
        applied = False
        for edit in edits:
            try:
                self.limits.check_files_touched(len(self.state.files_changed) + 1)
            except LimitExceeded as exc:
                self.state.errors.append(str(exc))
                break

            if edit.action == "write":
                result = self.tools.run(
                    "write_file",
                    {"path": edit.path, "content": edit.content or ""},
                )
                self.state.append_tool_result(
                    "write_file",
                    {"path": edit.path},
                    result.output if result.success else (result.error or ""),
                    success=result.success,
                )
                if result.success:
                    change_type = "created" if result.data.get("created") else "modified"
                    self.state.record_file_change(edit.path, change_type, edit.reason)
                    applied = True
                else:
                    self.state.errors.append(
                        f"write_file failed for {edit.path}: {result.error}"
                    )
            else:
                result = self.tools.run(
                    "apply_patch",
                    {
                        "path": edit.path,
                        "old_string": edit.old_string,
                        "new_string": edit.new_string,
                    },
                )
                self.state.append_tool_result(
                    "apply_patch",
                    {"path": edit.path},
                    result.output if result.success else (result.error or ""),
                    success=result.success,
                )
                if result.success:
                    self.state.record_file_change(edit.path, "modified", edit.reason)
                    applied = True
                else:
                    self.state.errors.append(
                        f"apply_patch failed for {edit.path}: {result.error}"
                    )
        return applied

    def stage_verify(self) -> None:
        """Verify syntax, feature presence, and preserved functionality."""
        logger.info("Stage: verify")
        if self._summary is None or self._plan is None:
            raise RuntimeError("Explore/plan required before verify")
        verifier = Verifier(self.state.repo_root, tools=self.tools)
        verification = verifier.verify(
            summary=self._summary,
            plan=self._plan,
            files_changed=self.state.files_changed,
            request=self.state.request,
        )
        self.state.verification = verification
        self.state.verify_notes = verification.as_notes()

        if self.agent_config.shell_enabled:
            for command in ("npm test", "pytest", "python -m pytest"):
                if command not in self.agent_config.shell_allowlist:
                    continue
                if command.startswith("npm") and not (
                    self.state.repo_root / "package.json"
                ).exists():
                    continue
                result = self.tools.run("execute_shell", {"command": command})
                self.state.append_tool_result(
                    "execute_shell",
                    {"command": command},
                    result.output if result.success else (result.error or ""),
                    success=result.success,
                )
                break

    def stage_summarize(self) -> None:
        """Produce the final report."""
        logger.info("Stage: summarize")
        if self._plan is None:
            raise RuntimeError("Plan required before summarize")
        verification = self.state.verification or Verifier(self.state.repo_root).verify(
            summary=self._summary,  # type: ignore[arg-type]
            plan=self._plan,
            files_changed=self.state.files_changed,
            request=self.state.request,
        )
        report = SummaryGenerator().generate(
            request=self.state.request,
            plan=self._plan,
            files_changed=self.state.files_changed,
            verification=verification,
        )
        self.state.final_report = report
        self.state.final_summary = report.overview
