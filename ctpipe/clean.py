"""clean subcommand: post-delivery cleanup of intermediate files.

Removes runs/ clone directories, optionally cleans ~/.claude/projects/ JSONL
trajectory cache, and optionally removes old delivery directories.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from ctpipe.config import BatchConfig, select_delivery_tasks
from ctpipe.project_hash import CLAUDE_PROJECTS_DIR, path_to_project_hash
from ctpipe.state import PipelineState


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def _dir_size(path: Path) -> int:
    total = 0
    try:
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
    except (OSError, PermissionError):
        pass
    return total


def clean(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    runs: bool = True,
    cache: bool = False,
    old_deliveries: bool = False,
    dry_run: bool = False,
) -> None:
    """Clean up intermediate files after delivery.

    Args:
        runs: Remove runs/ clone directories for current batch tasks.
        cache: Remove ~/.claude/projects/ JSONL trajectory cache for task run dirs.
        old_deliveries: Remove delivery_* directories other than the current one.
        dry_run: Only print what would be deleted.
    """
    tasks = select_delivery_tasks(config, task_ids)
    state = PipelineState(config.delivery_dir / "pipeline_state.json")

    removed_dirs: list[tuple[Path, int]] = []
    action = "Would remove" if dry_run else "Removing"

    # --- Clean runs/ directories ---
    if runs:
        print("=" * 60)
        print("Cleaning runs/ directories")
        print("=" * 60)

        dirs_to_remove: set[Path] = set()
        for task in tasks:
            prepare_info = state.get(task.id, "prepare")
            for model in ("qwen", "claude"):
                run_dir_str = prepare_info.get(f"{model}_dir", "")
                if run_dir_str:
                    run_dir = Path(run_dir_str)
                    task_root = run_dir.parent
                    if task_root.is_dir():
                        dirs_to_remove.add(task_root)

        if not dirs_to_remove and config.runs_root.is_dir():
            print(f"  WARNING: no task-specific dirs in state. Scanning runs_root for CT-* dirs...")
            for child in config.runs_root.iterdir():
                if child.is_dir() and child.name.startswith("CT-"):
                    dirs_to_remove.add(child)

        for d in sorted(dirs_to_remove):
            size = _dir_size(d)
            print(f"  {action}: {d} ({_human_size(size)})")
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)
            removed_dirs.append((d, size))

        if not dirs_to_remove:
            print("  (nothing to clean)")

    # --- Clean .claude/projects/ JSONL cache ---
    if cache:
        print("\n" + "=" * 60)
        print("Cleaning ~/.claude/projects/ trajectory cache")
        print("=" * 60)

        cache_dirs: set[Path] = set()
        for task in tasks:
            prepare_info = state.get(task.id, "prepare")
            for model in ("qwen", "claude"):
                run_dir_str = prepare_info.get(f"{model}_dir", "")
                if not run_dir_str:
                    continue
                try:
                    proj_hash = path_to_project_hash(run_dir_str)
                    proj_cache = CLAUDE_PROJECTS_DIR / proj_hash
                    if proj_cache.is_dir():
                        cache_dirs.add(proj_cache)
                except ValueError:
                    pass

        for d in sorted(cache_dirs):
            size = _dir_size(d)
            print(f"  {action}: {d} ({_human_size(size)})")
            if not dry_run:
                shutil.rmtree(d, ignore_errors=True)
            removed_dirs.append((d, size))

        if not cache_dirs:
            print("  (no cache directories found)")

    # --- Clean old delivery directories ---
    if old_deliveries:
        print("\n" + "=" * 60)
        print("Cleaning old delivery directories")
        print("=" * 60)

        current_delivery = config.delivery_dir.resolve()
        base_dir = config.base_dir
        for child in sorted(base_dir.iterdir()):
            if child.is_dir() and child.name.startswith("delivery_"):
                if child.resolve() != current_delivery:
                    size = _dir_size(child)
                    print(f"  {action}: {child} ({_human_size(size)})")
                    if not dry_run:
                        shutil.rmtree(child, ignore_errors=True)
                    removed_dirs.append((child, size))

    # --- Summary ---
    total_size = sum(size for _, size in removed_dirs)
    print("\n" + "-" * 60)
    if dry_run:
        print(f"Dry run: would remove {len(removed_dirs)} directories ({_human_size(total_size)})")
        print("Run without --dry-run to actually delete.")
    else:
        print(f"Cleaned {len(removed_dirs)} directories, freed {_human_size(total_size)}")
