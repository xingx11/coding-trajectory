"""Convert Windows absolute paths to Claude Code project hash directory names.

Verified mapping: D:\\A3Code\\YongFu → D--A3Code-YongFu
Note: spaces and hyphens both map to '-', which means paths differing
only in spaces vs hyphens will collide. This matches Claude Code's behavior.
"""

from __future__ import annotations

from pathlib import Path

CLAUDE_PROJECTS_DIR = Path.home() / ".claude" / "projects"


def path_to_project_hash(abs_path: str | Path) -> str:
    p = str(abs_path).replace("\\", "/").rstrip("/")
    drive, _, rest = p.partition(":/")
    if not drive or not rest:
        raise ValueError(f"Expected absolute Windows path, got: {abs_path}")
    slug = rest.replace("/", "-").replace(" ", "-")
    return f"{drive}--{slug}"


def project_hash_dir(abs_path: str | Path) -> Path:
    return CLAUDE_PROJECTS_DIR / path_to_project_hash(abs_path)
