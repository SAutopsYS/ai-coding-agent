"""Helpers for rendering a compact repository tree summary."""

from __future__ import annotations

from pathlib import Path

from agent.explorer.models import RepoNode, RepositoryMap


def format_tree_summary(
    repo_map: RepositoryMap,
    *,
    max_depth: int = 3,
    max_entries_per_dir: int = 40,
) -> str:
    """Render a compact textual tree from the repository map."""
    if repo_map.tree is None:
        root_name = Path(repo_map.root).name
        lines = [f"{root_name}/"]
        for directory in repo_map.directories[:50]:
            lines.append(f"  {directory}/")
        for file_path in repo_map.files[:80]:
            lines.append(f"  {file_path}")
        return "\n".join(lines)

    lines: list[str] = []
    _render_node(
        repo_map.tree,
        lines=lines,
        depth=0,
        prefix="",
        max_depth=max_depth,
        max_entries_per_dir=max_entries_per_dir,
        is_last=True,
        is_root=True,
    )
    return "\n".join(lines)


def _render_node(
    node: RepoNode,
    *,
    lines: list[str],
    depth: int,
    prefix: str,
    max_depth: int,
    max_entries_per_dir: int,
    is_last: bool,
    is_root: bool,
) -> None:
    label = f"{node.name}/" if node.kind == "directory" else node.name
    if is_root:
        lines.append(label)
        child_prefix = ""
    else:
        connector = "└── " if is_last else "├── "
        lines.append(f"{prefix}{connector}{label}")
        child_prefix = f"{prefix}{'    ' if is_last else '│   '}"

    if node.kind != "directory" or depth >= max_depth:
        return

    children = node.children
    visible = children[:max_entries_per_dir]
    for index, child in enumerate(visible):
        _render_node(
            child,
            lines=lines,
            depth=depth + 1,
            prefix=child_prefix,
            max_depth=max_depth,
            max_entries_per_dir=max_entries_per_dir,
            is_last=index == len(visible) - 1 and len(children) <= max_entries_per_dir,
            is_root=False,
        )

    remaining = len(children) - len(visible)
    if remaining > 0:
        lines.append(f"{child_prefix}└── ... ({remaining} more)")
