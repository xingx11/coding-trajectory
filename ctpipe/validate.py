"""Validation helpers for delivery completeness and naming consistency."""

from __future__ import annotations

import csv

from ctpipe.config import (
    BAD_PATTERNS,
    VALID_TASK_TYPES,
    BatchConfig,
    check_passrate_thresholds,
    model_stem,
    select_delivery_tasks,
)
from ctpipe.finalize import assign_submission_ids
from ctpipe.toml_utils import read_quality_toml, safe_calc_passrate
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory, trajectory_filename


def validate(config: BatchConfig, task_ids: list[str] | None = None, models: list[str] | None = None, *, dry_run: bool = False, as_json: bool = False) -> bool | dict:
    # Always compute submission IDs from the full task list so that
    # sequence numbers stay consistent whether or not --tasks is used.
    all_tasks = select_delivery_tasks(config, task_ids=None)
    submission_id_map = assign_submission_ids(all_tasks, config.person_id, config.delivery_date)

    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]
    delivery_dir = config.delivery_dir

    if dry_run:
        checks = [
            "metadata existence",
            "task_type / bad_pattern validity",
            "submission row presence & submission ID mapping",
            "trajectory existence & parse validity",
            "trajectory provider & session_id consistency",
            "score file existence & completeness",
            "passrate calculation & CSV consistency",
            "passrate threshold checks (qwen < 0.7, gap > 0.25)",
        ]

        tasks_data = []
        for task in tasks:
            sub_id = submission_id_map.get(task.id, task.id)
            meta = delivery_dir / "metadata" / f"{task.id}.md"
            models_info = {}
            for model_name in models:
                traj = config.resolve_trajectory_path(task.id, model_name)
                score = config.resolve_score_path(task.id, model_name)
                models_info[model_name] = {
                    "trajectory_exists": traj.exists(),
                    "score_exists": score.exists(),
                }
            tasks_data.append({
                "task_id": task.id, "submission_id": sub_id,
                "metadata_exists": meta.exists(),
                "models": models_info,
            })

        if as_json:
            return {
                "delivery_dir": str(delivery_dir),
                "submission_csv": str(delivery_dir / "submission.csv"),
                "submission_csv_exists": (delivery_dir / "submission.csv").exists(),
                "checks": checks,
                "tasks": tasks_data,
                "summary": {
                    "total": len(tasks),
                    "trajectories": len(tasks) * len(models),
                    "scores": len(tasks) * len(models),
                },
            }

        print("=" * 60)
        print("  DRY RUN: validate")
        print("=" * 60)
        print(f"Delivery: {delivery_dir.name}")
        print(f"Tasks:    {len(tasks)}")
        print(f"Models:   {', '.join(models)}")
        print(f"Submission CSV: {delivery_dir / 'submission.csv'}"
              f"{' (exists)' if (delivery_dir / 'submission.csv').exists() else ' (missing)'}")

        print(f"\nChecks to perform ({len(checks)}):")
        for check in checks:
            print(f"  - {check}")

        print(f"\nFiles to validate per task:")
        for t in tasks_data:
            print(f"\n  [{t['task_id']}] -> {t['submission_id']}")
            print(f"    metadata:  {'OK' if t['metadata_exists'] else 'MISSING'}")
            for model_name in models:
                mi = t["models"][model_name]
                print(f"    {model_name}:")
                print(f"      trajectory: {'OK' if mi['trajectory_exists'] else 'MISSING'}")
                print(f"      score:      {'OK' if mi['score_exists'] else 'MISSING'}")

        print(f"\nTotal: {len(tasks)} task(s), "
              f"{len(tasks) * len(models)} trajectory(ies), "
              f"{len(tasks) * len(models)} score file(s)")
        return True

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

        if task.task_type not in VALID_TASK_TYPES:
            issues.append(f"[{task.id}] invalid task_type: {task.task_type!r}")
        if task.bad_pattern and task.bad_pattern not in BAD_PATTERNS:
            issues.append(f"[{task.id}] invalid bad_pattern: {task.bad_pattern!r}")

        sub_id = submission_id_map.get(task.id, task.id)
        row = submission_rows.get(sub_id)
        if not row:
            issues.append(f"[{task.id}] submission row missing (expected id={sub_id})")

        # Cache passrates from score reads for threshold checking below
        cached_passrates: dict[str, float] = {}
        cached_criteria: dict[str, list] = {}  # for cross-model checks (A3-f)
        cached_first_queries: dict[str, str] = {}  # for B1-d query consistency

        for model_name in models:
            trajectory_path = find_delivery_trajectory(delivery_dir, model_name, task.id)
            if not trajectory_path:
                issues.append(
                    f"[{task.id}/{model_name}] trajectory missing: "
                    f"trajectories/{model_name}/{trajectory_filename(task.id, model_name)}"
                )
                continue

            try:
                info = parse_trajectory(trajectory_path)
            except Exception as exc:
                issues.append(
                    f"[{task.id}/{model_name}] trajectory parse error: {exc}"
                )
                continue
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
            if info.first_user_query:
                cached_first_queries[model_name] = info.first_user_query

            expected_rel = f"trajectories/{model_name}/{trajectory_filename(task.id, model_name)}"
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
                expected_new = f"scores/{model_name}/{model_stem(task.id, model_name)}.quality.toml"
                expected_legacy = f"scores/{model_name}/{task.id}.quality.toml"
                if row.get(score_key, "") not in (expected_new, expected_legacy):
                    issues.append(
                        f"[{task.id}/{model_name}] submission score path mismatch: "
                        f"{row.get(score_key, '')!r} != {expected_new!r}"
                    )

            score_path = config.resolve_score_path(task.id, model_name)
            pr, score_err = safe_calc_passrate(score_path)
            if pr is None:
                issues.append(f"[{task.id}/{model_name}] {score_err}")
                continue

            criteria = read_quality_toml(score_path)
            cached_passrates[model_name] = pr
            cached_criteria[model_name] = criteria
            passrate = f"{pr:.4f}"
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

        # B1-d: first user query must be identical between qwen and claude
        if len(cached_first_queries) == 2:
            q_query = cached_first_queries.get("qwen", "")
            c_query = cached_first_queries.get("claude", "")
            if q_query and c_query and q_query != c_query:
                issues.append(
                    f"[{task.id}] B1-d: first user query differs between qwen and claude"
                )

        # Threshold checks using cached passrates (no re-read needed)
        qwen_pr = cached_passrates.get("qwen", 0.0)
        claude_pr = cached_passrates.get("claude", 0.0)
        has_qwen = "qwen" in cached_passrates
        has_claude = "claude" in cached_passrates

        issues.extend(check_passrate_thresholds(
            task.id, qwen_pr, claude_pr, has_qwen, has_claude,
        ))

        # A3-f: at least 1 dimension with equal qwen/claude scores (non-bias evidence)
        if "qwen" in cached_criteria and "claude" in cached_criteria:
            qwen_scores = {c.name: c.score for c in cached_criteria["qwen"]}
            claude_scores = {c.name: c.score for c in cached_criteria["claude"]}
            shared_dims = set(qwen_scores) & set(claude_scores)
            if shared_dims and not any(qwen_scores[d] == claude_scores[d] for d in shared_dims):
                issues.append(
                    f"[{task.id}] A3-f: no dimension has equal qwen/claude scores — "
                    f"may indicate biased scoring"
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
