"""Runtime limits for an agent run."""

from __future__ import annotations

from dataclasses import dataclass, field


class LimitExceeded(RuntimeError):
    """Raised when a configured safety limit is exceeded."""


@dataclass
class Limits:
    """Numeric and policy caps for safety."""

    max_steps: int = 30
    max_files_touched: int = 20
    max_read_bytes: int = 200_000
    max_write_bytes: int = 200_000
    max_repair_attempts: int = 1
    forbidden_path_globs: list[str] = field(default_factory=list)

    def check_step_count(self, step_count: int) -> None:
        """Raise if ``step_count`` exceeds ``max_steps``."""
        if step_count > self.max_steps:
            raise LimitExceeded(f"Exceeded max_steps={self.max_steps}")

    def check_read_size(self, num_bytes: int) -> None:
        """Raise if a read exceeds ``max_read_bytes``."""
        if num_bytes > self.max_read_bytes:
            raise LimitExceeded(
                f"Read size {num_bytes} exceeds max_read_bytes={self.max_read_bytes}"
            )

    def check_write_size(self, num_bytes: int) -> None:
        """Raise if a write exceeds ``max_write_bytes``."""
        if num_bytes > self.max_write_bytes:
            raise LimitExceeded(
                f"Write size {num_bytes} exceeds max_write_bytes={self.max_write_bytes}"
            )

    def check_files_touched(self, count: int) -> None:
        """Raise if too many files were modified."""
        if count > self.max_files_touched:
            raise LimitExceeded(
                f"Touched {count} files; max_files_touched={self.max_files_touched}"
            )
