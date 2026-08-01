"""Path containment under the repository root."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sandbox:
    """Resolves and validates paths against a single repo root."""

    repo_root: Path

    def __post_init__(self) -> None:
        self.repo_root = self.repo_root.resolve()

    def resolve(self, relative_path: str | Path) -> Path:
        """Resolve ``relative_path`` to an absolute path inside the repo.

        Raises:
            ValueError: If the path escapes the repository root.
        """
        raw = Path(relative_path)
        if raw.is_absolute():
            candidate = raw.resolve()
        else:
            candidate = (self.repo_root / raw).resolve()

        if not self.is_inside(candidate):
            raise ValueError(f"Path escapes repository root: {relative_path}")
        return candidate

    def is_inside(self, path: Path) -> bool:
        """Return True if ``path`` is inside ``repo_root``."""
        try:
            path.resolve().relative_to(self.repo_root)
            return True
        except ValueError:
            return False

    def to_relative(self, path: Path) -> str:
        """Return a posix relative path for ``path`` under the repo root."""
        return path.resolve().relative_to(self.repo_root).as_posix()
