"""prepare subcommand: clone projects and create delivery directory skeleton."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

from ctpipe.config import BatchConfig, TaskConfig, select_tasks
from ctpipe.state import PipelineState

SETTINGS_LOCAL = {
    "permissions": {
        "allow": [
            "Bash(*)",
            "Read(*)",
            "Edit(*)",
            "Write(*)",
            "Glob(*)",
            "Grep(*)",
            "NotebookEdit(*)",
        ]
    }
}

IGNORE_PATTERNS = shutil.ignore_patterns(
    "node_modules", ".venv", "__pycache__", ".git", "dist", ".next", ".nuxt",
)


def _clone_project(task: TaskConfig, model: str, runs_root: Path) -> Path:
    dest = runs_root / f"{task.id}-{model}" / task.project_subdir
    if dest.exists():
        print(f"  [skip] {dest} already exists")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    src = task.project_path

    # Check if the other model's clone already exists — reuse it as local source
    other_model = "claude" if model == "qwen" else "qwen"
    other_dest = runs_root / f"{task.id}-{other_model}" / task.project_subdir

    if other_dest.is_dir() and task.clone_method == "git" and (other_dest / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--local", "--depth", "1", str(other_dest), str(dest)],
            check=True,
            capture_output=True,
        )
        print(f"  [local clone] {other_dest.name} -> {dest}")
    elif other_dest.is_dir():
        shutil.copytree(str(other_dest), str(dest), ignore=IGNORE_PATTERNS)
        print(f"  [copy from sibling] {other_dest.name} -> {dest}")
    elif task.clone_method == "git" and (src / ".git").is_dir():
        subprocess.run(
            ["git", "clone", "--depth", "1", str(src), str(dest)],
            check=True,
            capture_output=True,
        )
        print(f"  [git clone] {src} -> {dest}")
    else:
        shutil.copytree(str(src), str(dest), ignore=IGNORE_PATTERNS)
        print(f"  [copy] {src} -> {dest}")

    claude_dir = dest / ".claude"
    claude_dir.mkdir(parents=True, exist_ok=True)
    (claude_dir / "settings.local.json").write_text(
        json.dumps(SETTINGS_LOCAL, indent=2),
        encoding="utf-8",
    )
    return dest


def _create_submission_csv(config: BatchConfig, csv_path: Path) -> None:
    template_csv = config.docs_dir / "submission_template.csv"
    if template_csv.exists():
        shutil.copy2(template_csv, csv_path)
        return

    fieldnames = [
        "id",
        "qwen 本地trajectory", "qwen session id", "qwen rubrics 人工评分",
        "claude 本地trajectory", "claude session id", "claude rubrics 人工评分",
        "qwen passrate", "claude passrate",
        "任务类型", "应用领域", "编程语言",
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _create_delivery_skeleton(config: BatchConfig) -> None:
    delivery_dir = config.delivery_dir
    for sub in [
        "trajectories/qwen",
        "trajectories/claude",
        "scores/qwen",
        "scores/claude",
        "metadata",
    ]:
        (delivery_dir / sub).mkdir(parents=True, exist_ok=True)

    for task in config.tasks:
        for model in ("qwen", "claude"):
            src_template = config.rubrics_dir / model / f"{task.id}.quality.toml"
            dest_score = delivery_dir / "scores" / model / f"{task.id}.quality.toml"
            if src_template.exists() and not dest_score.exists():
                shutil.copy2(src_template, dest_score)

    csv_path = delivery_dir / "submission.csv"
    if not csv_path.exists():
        _create_submission_csv(config, csv_path)


def prepare(config: BatchConfig, task_ids: list[str] | None = None) -> None:
    state = PipelineState(config.delivery_dir / "pipeline_state.json")

    print("Creating delivery directory skeleton...")
    _create_delivery_skeleton(config)

    tasks = select_tasks(config.tasks, task_ids)

    for task in tasks:
        if state.is_done(task.id, "prepare"):
            print(f"[{task.id}] prepare already done, skipping")
            continue

        print(f"[{task.id}] Cloning project for qwen and claude...")
        qwen_dir = _clone_project(task, "qwen", config.runs_root)
        claude_dir = _clone_project(task, "claude", config.runs_root)

        state.set(
            task.id,
            "prepare",
            status="done",
            qwen_dir=str(qwen_dir),
            claude_dir=str(claude_dir),
        )

    print("Prepare complete.")
