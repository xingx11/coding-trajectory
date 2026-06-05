"""prepare subcommand: clone projects and create delivery directory skeleton."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

from ctpipe.config import SUBMISSION_FIELDNAMES, BatchConfig, TaskConfig, load_task_manifest, select_tasks, write_task_manifest
from ctpipe.state import PipelineState


def _all_known_tasks(config: BatchConfig) -> list[TaskConfig]:
    manifest = load_task_manifest(config.task_manifest_path)
    seen: dict[str, TaskConfig] = {t.id: t for t in manifest}
    for t in config.tasks:
        if t.id not in seen:
            seen[t.id] = t
    return list(seen.values())

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

    if task.clone_method == "git" and (src / ".git").is_dir():
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
        with template_csv.open("r", encoding="utf-8-sig", newline="") as src:
            reader = csv.reader(src)
            header = next(reader, None)
        if header:
            with csv_path.open("w", encoding="utf-8-sig", newline="") as dest:
                writer = csv.writer(dest)
                writer.writerow(header)
        return

    fieldnames = SUBMISSION_FIELDNAMES
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

    for task in _all_known_tasks(config):
        for model in ("qwen", "claude"):
            src_template = config.rubrics_dir / model / f"{task.id}.quality.toml"
            dest_score = delivery_dir / "scores" / model / f"{task.id}.quality.toml"
            if src_template.exists() and not dest_score.exists():
                shutil.copy2(src_template, dest_score)

    csv_path = delivery_dir / "submission.csv"
    if not csv_path.exists():
        _create_submission_csv(config, csv_path)


def _build_metadata_content(task: TaskConfig) -> str:
    followups_qwen = "\n".join(f"- {item}" for item in task.followups_qwen) or "- "
    followups_claude = "\n".join(f"- {item}" for item in task.followups_claude) or "- "
    return f"""# {task.id} Metadata

## Codebase

- Project path: {task.project_path}
- Source: local project / open-source project
- Open-source repo URL:
- Commit / branch / snapshot:

## Task Label

- Task type: {task.task_type}
- Application domain: {task.domain}
- Language: {task.language}

## Qwen Conversation

- Session id:
- Trajectory file: trajectories/qwen/{task.id}.jsonl
- Round count: {1 + len(task.followups_qwen)}
- Prompt strategy: same-theme / different-follow-up / related-task
- Initial prompt:

```text
{task.prompt_qwen}
```

- Follow-up summary:

```text
{followups_qwen}
```

## Claude Conversation

- Session id:
- Trajectory file: trajectories/claude/{task.id}.jsonl
- Round count: {1 + len(task.followups_claude)}
- Prompt strategy: same-theme / different-follow-up / related-task
- Initial prompt:

```text
{task.prompt_claude}
```

- Follow-up summary:

```text
{followups_claude}
```

## Scoring

- Qwen score file: scores/qwen/{task.id}.quality.toml
- Claude score file: scores/claude/{task.id}.quality.toml
- Qwen passrate:
- Claude passrate:

## Notes

- Are Qwen and Claude prompts identical? yes / no
- If no, why are they still comparable?
- Any environment issue:
- Any manual intervention:
"""


def _create_metadata_stub(config: BatchConfig, task: TaskConfig) -> None:
    metadata_path = config.delivery_dir / "metadata" / f"{task.id}.md"
    if metadata_path.exists():
        return
    metadata_path.write_text(_build_metadata_content(task), encoding="utf-8")


def prepare(config: BatchConfig, task_ids: list[str] | None = None) -> None:
    state = PipelineState(config.delivery_dir / "pipeline_state.json")

    print("Creating delivery directory skeleton...")
    _create_delivery_skeleton(config)

    all_tasks = _all_known_tasks(config)
    tasks = select_tasks(all_tasks, task_ids)

    existing = load_task_manifest(config.task_manifest_path)
    existing_ids = {t.id for t in existing}
    merged = list(existing)
    for task in tasks:
        if task.id not in existing_ids:
            merged.append(task)
            existing_ids.add(task.id)
    write_task_manifest(config.task_manifest_path, merged)

    for task in tasks:
        with state.batch():
            if state.is_done(task.id, "prepare"):
                prep_info = state.get(task.id, "prepare")
                qwen_ok = Path(prep_info.get("qwen_dir", "")).is_dir()
                claude_ok = Path(prep_info.get("claude_dir", "")).is_dir()
                if qwen_ok and claude_ok:
                    print(f"[{task.id}] prepare already done, skipping")
                    _create_metadata_stub(config, task)
                    continue
                print(f"[{task.id}] prepare marked done but directories missing, re-cloning...")

            print(f"[{task.id}] Cloning project for qwen and claude...")
            qwen_dir = _clone_project(task, "qwen", config.runs_root)
            claude_dir = _clone_project(task, "claude", config.runs_root)
            _create_metadata_stub(config, task)

            state.set(
                task.id,
                "prepare",
                status="done",
                qwen_dir=str(qwen_dir),
                claude_dir=str(claude_dir),
            )

    print("Prepare complete.")
