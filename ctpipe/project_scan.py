"""Lightweight project scanning utilities.

Used by gen.py, prepare.py, score.py, and rescore.py to build concise
project summaries without pulling in heavy gen.py dependencies
(GitHub/Gitee search, distribution tables, etc.).
"""

from __future__ import annotations

import functools
from pathlib import Path

SCAN_IGNORE = {
    "node_modules", ".venv", "__pycache__", ".git", "dist", ".next", ".nuxt",
    "build", "target", ".gradle", ".idea", ".vscode", "vendor", "coverage",
    ".tox", "eggs", ".mypy_cache", ".pytest_cache",
}

SCAN_IGNORE_SUFFIXES = {".egg-info"}


@functools.lru_cache(maxsize=64)
def scan_project(project_path: Path, max_chars: int = 1500) -> str:
    """Build a concise project summary: README excerpt + tree + deps."""
    parts: list[str] = []

    for readme_name in ("README.md", "readme.md", "README.rst", "README"):
        readme_path = project_path / readme_name
        if readme_path.is_file() and not readme_path.is_symlink():
            try:
                content = readme_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()[:20]
                parts.append(f"## README\n" + "\n".join(lines) + "\n")
            except Exception:
                pass
            break

    tree_lines: list[str] = []
    _walk_tree(project_path, "", tree_lines, depth=0, max_depth=2, max_lines=40)
    parts.append("## Tree\n" + "\n".join(tree_lines[:40]) + "\n")

    dep_files = ["package.json", "pyproject.toml", "Cargo.toml", "go.mod", "pom.xml",
                 "build.gradle", "Gemfile", "composer.json"]
    for dep_name in dep_files:
        dep_path = project_path / dep_name
        if dep_path.is_file() and not dep_path.is_symlink():
            try:
                content = dep_path.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()[:15]
                parts.append(f"## {dep_name}\n" + "\n".join(lines) + "\n")
            except Exception:
                pass
            break

    result = "\n".join(parts)
    if len(result) > max_chars:
        result = result[:max_chars] + "\n[... truncated ...]"
    return result


def _should_ignore(name: str) -> bool:
    if name in SCAN_IGNORE or name.startswith("."):
        return True
    return any(name.endswith(s) for s in SCAN_IGNORE_SUFFIXES)


def _walk_tree(
    path: Path, prefix: str, lines: list[str],
    depth: int = 0, max_depth: int = 4, max_lines: int = 200,
) -> None:
    if depth > max_depth or len(lines) >= max_lines:
        return
    try:
        entries = sorted(path.iterdir(), key=lambda e: (not e.is_dir(), e.name.lower()))
    except PermissionError:
        return

    for entry in entries:
        if _should_ignore(entry.name):
            continue
        if entry.is_symlink():
            continue
        if len(lines) >= max_lines:
            return
        lines.append(f"{prefix}{entry.name}{'/' if entry.is_dir() else ''}")
        if entry.is_dir():
            _walk_tree(entry, prefix + "  ", lines, depth + 1, max_depth, max_lines)
