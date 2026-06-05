"""Validation helpers for delivery completeness and naming consistency."""

from __future__ import annotations

import csv

from ctpipe.config import BatchConfig, select_delivery_tasks
from ctpipe.toml_utils import calc_passrate, is_complete_rubric, is_unscored_template, read_quality_toml
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory, trajectory_filename


def validate(config: BatchConfig, task_ids: list[str] | None = None, models: list[str] | None = None) -> bool:
    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]
    delivery_dir = config.delivery_dir
    issues: list[str] = []

    if not tasks:
        issues.append("no tasks found in delivery manifest or tasks.toml")

    submission_rows: dict[str, dict[str, str]] = {}
    submission_path = delivery_dir / "submission.csv"
    if submission_path.exists():
        with submission_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id"):
                    submission_rows[row["id"]] = row
    else:
        issues.append("submission.csv is missing")

    for task in tasks:
        metadata_path = delivery_dir / "metadata" / f"{task.id}.md"
        if not metadata_path.exists():
            issues.append(f"[{task.id}] metadata missing: {metadata_path.name}")

        row = submission_rows.get(task.id)
        if not row:
            issues.append(f"[{task.id}] submission row missing")

        for model_name in models:
            trajectory_path = find_delivery_trajectory(delivery_dir, model_name, task.id)
            if not trajectory_path:
                issues.append(
                    f"[{task.id}/{model_name}] trajectory missing: "
                    f"trajectories/{model_name}/{trajectory_filename(task.id)}"
                )
                continue

            info = parse_trajectory(trajectory_path)
            if not info.session_id and not info.models:
                issues.append(
                    f"[{task.id}/{model_name}] trajectory has no valid content "
                    f"(lines={info.line_count})"
                )
                continue
            if info.detected_provider not in ("unknown", model_name):
                issues.append(
                    f"[{task.id}/{model_name}] provider mismatch: detected {info.detected_provider}"
                )

            expected_rel = f"trajectories/{model_name}/{trajectory_filename(task.id)}"
            if row:
                rel_key = f"{model_name} 本地trajectory"
                if row.get(rel_key, "") != expected_rel:
                    issues.append(
                        f"[{task.id}/{model_name}] submission trajectory mismatch: "
                        f"{row.get(rel_key, '')!r} != {expected_rel!r}"
                    )

                session_key = f"{model_name} session id"
                if row.get(session_key, "") != info.session_id:
                    issues.append(
                        f"[{task.id}/{model_name}] submission session mismatch: "
                        f"{row.get(session_key, '')!r} != {info.session_id!r}"
                    )

                score_key = f"{model_name} rubrics 人工评分"
                expected_score = f"scores/{model_name}/{task.id}.quality.toml"
                if row.get(score_key, "") != expected_score:
                    issues.append(
                        f"[{task.id}/{model_name}] submission score path mismatch: "
                        f"{row.get(score_key, '')!r} != {expected_score!r}"
                    )

            score_path = delivery_dir / "scores" / model_name / f"{task.id}.quality.toml"
            if not score_path.exists():
                issues.append(f"[{task.id}/{model_name}] score file missing: {score_path.name}")
                continue

            try:
                criteria = read_quality_toml(score_path)
            except Exception as exc:
                issues.append(f"[{task.id}/{model_name}] score read error: {exc}")
                continue

            if is_unscored_template(criteria):
                issues.append(f"[{task.id}/{model_name}] score file is unscored template: {score_path.name}")
                continue

            if not is_complete_rubric(criteria):
                issues.append(
                    f"[{task.id}/{model_name}] score file incomplete: "
                    f"{len(criteria)} criteria (expected 7)"
                )
                continue

            passrate = f"{calc_passrate(criteria):.4f}"
            if row and row.get(f"{model_name} passrate", "") != passrate:
                issues.append(
                    f"[{task.id}/{model_name}] submission passrate mismatch: "
                    f"{row.get(f'{model_name} passrate', '')!r} != {passrate!r}"
                )

    print("\nValidation summary")
    print("=" * 60)
    print(f"Delivery directory: {delivery_dir}")
    print(f"Tasks checked: {len(tasks)}")
    if issues:
        print(f"Issues: {len(issues)}")
        for issue in issues:
            print(f"  - {issue}")
        return False

    print("No issues found.")
    return True
