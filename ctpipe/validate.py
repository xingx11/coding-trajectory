"""Validation helpers for delivery completeness and naming consistency."""

from __future__ import annotations

import csv

from ctpipe.config import (
    BAD_PATTERNS,
    THRESHOLD_CLAUDE_MIN,
    THRESHOLD_QWEN_MAX,
    THRESHOLD_RELATIVE_GAIN_MIN,
    VALID_TASK_TYPES,
    BatchConfig,
    select_delivery_tasks,
)
from ctpipe.finalize import _assign_submission_ids
from ctpipe.toml_utils import calc_passrate, is_complete_rubric, is_unscored_template, read_quality_toml
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory, trajectory_filename


def validate(config: BatchConfig, task_ids: list[str] | None = None, models: list[str] | None = None) -> bool:
    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]
    delivery_dir = config.delivery_dir
    issues: list[str] = []

    if not tasks:
        issues.append("no tasks found in delivery manifest or tasks.toml")

    # Build submission ID mapping to match CSV rows
    submission_id_map = _assign_submission_ids(tasks, config.person_id, config.delivery_date)

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

        if task.task_type not in VALID_TASK_TYPES:
            issues.append(f"[{task.id}] invalid task_type: {task.task_type!r}")
        if task.bad_pattern and task.bad_pattern not in BAD_PATTERNS:
            issues.append(f"[{task.id}] invalid bad_pattern: {task.bad_pattern!r}")

        sub_id = submission_id_map.get(task.id, task.id)
        row = submission_rows.get(sub_id)
        if not row:
            issues.append(f"[{task.id}] submission row missing (expected id={sub_id})")

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
            actual_rel = trajectory_path.relative_to(delivery_dir).as_posix()
            if row:
                rel_key = f"{model_name} 本地trajectory"
                row_rel = row.get(rel_key, "")
                if row_rel and row_rel != actual_rel and row_rel != expected_rel:
                    issues.append(
                        f"[{task.id}/{model_name}] submission trajectory mismatch: "
                        f"{row_rel!r} != {actual_rel!r}"
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
                scored_count = sum(1 for c in criteria if c.score > 0 or c.rationale)
                issues.append(
                    f"[{task.id}/{model_name}] score file incomplete: "
                    f"{scored_count}/7 criteria scored"
                )
                continue

            passrate = f"{calc_passrate(criteria):.4f}"
            if row:
                csv_pr = row.get(f"{model_name} passrate", "")
                if csv_pr:
                    try:
                        if abs(float(csv_pr) - float(passrate)) > 0.0001:
                            issues.append(
                                f"[{task.id}/{model_name}] submission passrate mismatch: "
                                f"{csv_pr!r} != {passrate!r}"
                            )
                    except ValueError:
                        issues.append(
                            f"[{task.id}/{model_name}] submission passrate mismatch: "
                            f"{csv_pr!r} != {passrate!r}"
                        )

        qwen_pr = 0.0
        claude_pr = 0.0
        has_qwen = False
        has_claude = False
        for model_name in models:
            score_path = delivery_dir / "scores" / model_name / f"{task.id}.quality.toml"
            if score_path.exists():
                try:
                    criteria = read_quality_toml(score_path)
                    if is_complete_rubric(criteria) and not is_unscored_template(criteria):
                        pr = calc_passrate(criteria)
                        if model_name == "qwen":
                            qwen_pr = pr
                            has_qwen = True
                        elif model_name == "claude":
                            claude_pr = pr
                            has_claude = True
                except Exception:
                    pass

        if has_qwen and qwen_pr >= THRESHOLD_QWEN_MAX:
            issues.append(f"[{task.id}] qwen passrate {qwen_pr:.4f} >= {THRESHOLD_QWEN_MAX}")
        if has_claude and has_qwen and claude_pr <= qwen_pr:
            issues.append(f"[{task.id}] claude passrate {claude_pr:.4f} <= qwen {qwen_pr:.4f}")
        if has_claude and has_qwen:
            if qwen_pr > 0:
                relative_gain = (claude_pr - qwen_pr) / qwen_pr
                if relative_gain <= THRESHOLD_RELATIVE_GAIN_MIN:
                    issues.append(
                        f"[{task.id}] relative gain {relative_gain:.2%} <= {THRESHOLD_RELATIVE_GAIN_MIN:.0%} "
                        f"(claude={claude_pr:.4f}, qwen={qwen_pr:.4f})"
                    )
            else:
                if claude_pr < THRESHOLD_RELATIVE_GAIN_MIN:
                    issues.append(
                        f"[{task.id}] qwen=0, claude passrate {claude_pr:.4f} too low "
                        f"(need >= {THRESHOLD_RELATIVE_GAIN_MIN} when qwen=0)"
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
