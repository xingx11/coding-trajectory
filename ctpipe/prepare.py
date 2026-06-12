"""prepare subcommand: clone projects and create delivery directory skeleton."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from pathlib import Path

from ctpipe.config import SUBMISSION_FIELDNAMES, BatchConfig, TaskConfig, load_task_manifest, model_stem, select_tasks, write_task_manifest
from ctpipe.project_scan import SCAN_IGNORE
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

IGNORE_PATTERNS = shutil.ignore_patterns(*SCAN_IGNORE)


def _clone_project(task: TaskConfig, model: str, runs_root: Path) -> tuple[Path, str]:
    """Clone project for a model. Returns (dest_path, commit_hash)."""
    dest = runs_root / f"{task.id}-{model}" / task.project_subdir
    commit_hash = ""

    if dest.exists():
        print(f"  [skip] {dest} already exists")
        # Try to read commit hash from existing clone
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(dest), capture_output=True, text=True,
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
        except OSError:
            pass
        return dest, commit_hash

    dest.parent.mkdir(parents=True, exist_ok=True)
    src = task.project_path

    # Record source commit hash before cloning
    if (src / ".git").is_dir():
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=str(src), capture_output=True, text=True,
            )
            if result.returncode == 0:
                commit_hash = result.stdout.strip()
        except OSError:
            pass

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
    return dest, commit_hash


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
            dest_score = config.score_path(task.id, model)
            if src_template.exists() and not dest_score.exists():
                shutil.copy2(src_template, dest_score)

    csv_path = delivery_dir / "submission.csv"
    if not csv_path.exists():
        _create_submission_csv(config, csv_path)


def _build_metadata_content(task: TaskConfig, commit_hash: str = "") -> str:
    followups_qwen = "\n".join(f"- {item}" for item in task.followups_qwen) or "- "
    followups_claude = "\n".join(f"- {item}" for item in task.followups_claude) or "- "

    # Generate project summary from source code
    project_summary = ""
    if task.project_path.is_dir():
        from ctpipe.project_scan import scan_project
        project_summary = scan_project(task.project_path)

    # Build task description section if available
    task_desc_section = ""
    if task.task_title or task.task_description or task.acceptance_criteria:
        desc_parts = ["## Task Description\n"]
        if task.task_title:
            desc_parts.append(f"- Title: {task.task_title}")
        if task.task_description:
            desc_parts.append(f"- Description: {task.task_description}")
        if task.acceptance_criteria:
            criteria_lines = "\n".join(f"  - {item}" for item in task.acceptance_criteria)
            desc_parts.append(f"- Acceptance criteria:\n{criteria_lines}")
        task_desc_section = "\n".join(desc_parts) + "\n\n"

    return f"""# {task.id} Metadata

## Codebase

- Project path: {task.project_path}
- Source: local project / open-source project
- Open-source repo URL:
- Commit / branch / snapshot: {commit_hash}

## Project Summary

```text
{project_summary}
```

## Task Label

- Task type: {task.task_type}
- Application domain: {task.domain}
- Language: {task.language}

{task_desc_section}## Qwen Conversation

- Session id:
- Trajectory file: trajectories/qwen/{model_stem(task.id, 'qwen')}.jsonl
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
- Trajectory file: trajectories/claude/{model_stem(task.id, 'claude')}.jsonl
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

- Qwen score file: scores/qwen/{model_stem(task.id, 'qwen')}.quality.toml
- Claude score file: scores/claude/{model_stem(task.id, 'claude')}.quality.toml
- Qwen passrate:
- Claude passrate:

## Notes

- Are Qwen and Claude prompts identical? yes / no
- If no, why are they still comparable?
- Any environment issue:
- Any manual intervention:
"""


def _create_metadata_stub(config: BatchConfig, task: TaskConfig, commit_hash: str = "") -> None:
    metadata_path = config.delivery_dir / "metadata" / f"{task.id}.md"
    if metadata_path.exists():
        return
    metadata_path.write_text(_build_metadata_content(task, commit_hash), encoding="utf-8")


def prepare(config: BatchConfig, task_ids: list[str] | None = None, *, dry_run: bool = False, as_json: bool = False) -> dict | None:
    all_tasks = _all_known_tasks(config)
    tasks = select_tasks(all_tasks, task_ids)

    if dry_run:
        # Data tracking for JSON output
        skeleton_dirs = [
            "trajectories/qwen", "trajectories/claude",
            "scores/qwen", "scores/claude", "metadata",
        ]
        skeleton_data = [
            {"path": sub, "status": "exists" if (config.delivery_dir / sub).is_dir() else "will_create"}
            for sub in skeleton_dirs
        ]
        tasks_data = []

        if not as_json:
            print("=" * 60)
            print("  DRY RUN: prepare")
            print("=" * 60)
            print(f"\nDelivery skeleton: {config.delivery_dir}")
            for sub in skeleton_dirs:
                p = config.delivery_dir / sub
                status = "exists" if p.is_dir() else "will create"
                print(f"  {sub}/  [{status}]")

        state = PipelineState(config.state_path)
        will_clone = 0
        will_skip = 0
        for task in tasks:
            src = task.project_path
            is_git = task.clone_method == "git" and (src / ".git").is_dir()
            method = "git clone --depth 1" if is_git else "copytree"

            # Read source commit hash
            commit = ""
            if (src / ".git").is_dir():
                try:
                    r = subprocess.run(
                        ["git", "rev-parse", "HEAD"],
                        cwd=str(src), capture_output=True, text=True,
                    )
                    if r.returncode == 0:
                        commit = r.stdout.strip()[:12]
                except OSError:
                    pass

            models_data = {}
            for model in ("qwen", "claude"):
                dest = config.runs_root / f"{task.id}-{model}" / task.project_subdir
                models_data[model] = {"dest": str(dest), "exists": dest.exists()}

            # Check if already done
            done = state.is_done(task.id, "prepare")
            if done:
                prep_info = state.get(task.id, "prepare")
                qwen_ok = Path(prep_info.get("qwen_dir", "")).is_dir()
                claude_ok = Path(prep_info.get("claude_dir", "")).is_dir()
                if qwen_ok and claude_ok:
                    will_skip += 1
                    tasks_data.append({
                        "task_id": task.id, "skip": True,
                        "source": str(src), "commit": commit,
                        "method": method, "models": models_data,
                    })
                    if not as_json:
                        print(f"\n[{task.id}]  SKIP (already done)")
                        print(f"  source: {src}")
                        if commit:
                            print(f"  commit: {commit}")
                        for model in ("qwen", "claude"):
                            print(f"  {model}: {config.runs_root / f'{task.id}-{model}' / task.project_subdir}")
                    continue

            will_clone += 1
            tasks_data.append({
                "task_id": task.id, "skip": False,
                "source": str(src), "commit": commit,
                "method": method, "models": models_data,
            })
            if not as_json:
                print(f"\n[{task.id}]  CLONE")
                print(f"  source: {src}")
                if commit:
                    print(f"  commit: {commit}")
                print(f"  method: {method}")
                for model in ("qwen", "claude"):
                    dest = config.runs_root / f"{task.id}-{model}" / task.project_subdir
                    exists = dest.exists()
                    status = " (exists — will skip)" if exists else ""
                    print(f"  {model}: {dest}{status}")

        if as_json:
            return {
                "delivery_dir": str(config.delivery_dir),
                "skeleton": skeleton_data,
                "tasks": tasks_data,
                "summary": {"total": len(tasks), "to_clone": will_clone, "skipped": will_skip},
            }

        print(f"\nTotal: {len(tasks)} task(s), "
              f"{will_clone} to clone, {will_skip} already done")
        return

    state = PipelineState(config.state_path)

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
                    _create_metadata_stub(config, task, prep_info.get("commit_hash", ""))
                    continue
                print(f"[{task.id}] prepare marked done but directories missing, re-cloning...")

            print(f"[{task.id}] Cloning project for qwen and claude...")
            try:
                qwen_dir, qwen_hash = _clone_project(task, "qwen", config.runs_root)
                claude_dir, claude_hash = _clone_project(task, "claude", config.runs_root)
            except (subprocess.CalledProcessError, OSError) as exc:
                print(f"[{task.id}] ERROR: clone failed: {exc}")
                state.set(task.id, "prepare", status="failed", error=str(exc))
                continue

            commit_hash = qwen_hash or claude_hash
            if qwen_hash and claude_hash and qwen_hash != claude_hash:
                print(f"  [WARN] commit hash mismatch: qwen={qwen_hash[:8]} claude={claude_hash[:8]}")
            _create_metadata_stub(config, task, commit_hash)

            state.set(
                task.id,
                "prepare",
                status="done",
                qwen_dir=str(qwen_dir),
                claude_dir=str(claude_dir),
                commit_hash=commit_hash,
            )

    print("Prepare complete.")
