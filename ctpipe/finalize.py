"""finalize subcommand: calculate passrates and generate submission.csv."""

from __future__ import annotations

import csv
from pathlib import Path

from ctpipe.config import SUBMISSION_FIELDNAMES, SUBMISSION_KEY_MAP, BatchConfig, select_delivery_tasks
from ctpipe.state import PipelineState
from ctpipe.toml_utils import calc_passrate, is_complete_rubric, is_unscored_template, read_quality_toml
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory, trajectory_filename


def finalize(config: BatchConfig, task_ids: list[str] | None = None, models: list[str] | None = None) -> None:
    state = PipelineState(config.delivery_dir / "pipeline_state.json")
    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]

    if not tasks:
        print("No tasks found in delivery manifest or tasks.toml; nothing to finalize.")
        return

    rows: list[dict[str, str]] = []
    issues: list[str] = []

    with state.batch():
        for task in tasks:
            row: dict[str, str] = {"id": task.id}

            for model_name in models:
                score_path = config.delivery_dir / "scores" / model_name / f"{task.id}.quality.toml"
                collect_info = state.get(task.id, "collect", model_name)
                run_info = state.get(task.id, "run", model_name)

                session_id = collect_info.get("session_id", run_info.get("session_id", ""))
                turns = str(run_info.get("turns", ""))
                rel_jsonl_path = collect_info.get(
                    "jsonl_path",
                    f"trajectories/{model_name}/{trajectory_filename(task.id)}",
                )

                jsonl_file = find_delivery_trajectory(
                    config.delivery_dir,
                    model_name,
                    task.id,
                    session_id=session_id or None,
                )
                if jsonl_file:
                    rel_jsonl_path = jsonl_file.relative_to(config.delivery_dir).as_posix()
                    try:
                        traj_info = parse_trajectory(jsonl_file)
                        if not traj_info.session_id and not traj_info.models:
                            issues.append(
                                f"[{task.id}/{model_name}] trajectory has no valid content "
                                f"(lines={traj_info.line_count})"
                            )
                        else:
                            session_id = traj_info.session_id or session_id
                            if traj_info.detected_provider not in ("unknown", model_name):
                                issues.append(
                                    f"[{task.id}/{model_name}] provider mismatch: {traj_info.detected_provider}"
                                )
                    except Exception as exc:
                        issues.append(f"[{task.id}/{model_name}] trajectory parse error: {exc}")

                passrate = ""
                if score_path.exists():
                    try:
                        criteria = read_quality_toml(score_path)
                        if is_unscored_template(criteria):
                            issues.append(f"[{task.id}/{model_name}] score file is unscored template")
                        elif not is_complete_rubric(criteria):
                            issues.append(
                                f"[{task.id}/{model_name}] score file incomplete: "
                                f"{len(criteria)} criteria (expected 7)"
                            )
                        else:
                            passrate = f"{calc_passrate(criteria):.4f}"
                    except Exception as exc:
                        issues.append(f"[{task.id}/{model_name}] score read error: {exc}")
                else:
                    issues.append(f"[{task.id}/{model_name}] missing score file")

                prefix = model_name
                row[f"{prefix}_trajectory"] = rel_jsonl_path
                row[f"{prefix}_session_id"] = session_id
                row[f"{prefix}_score_path"] = f"scores/{model_name}/{task.id}.quality.toml"
                row[f"{prefix}_passrate"] = passrate
                row[f"{prefix}_turns"] = turns

            row["task_type"] = task.task_type
            row["domain"] = task.domain
            row["language"] = task.language

            qwen_pr = float(row.get("qwen_passrate") or 0)
            claude_pr = float(row.get("claude_passrate") or 0)
            has_qwen = bool(row.get("qwen_passrate"))
            has_claude = bool(row.get("claude_passrate"))

            task_prefix = f"[{task.id}]"
            task_slash = f"[{task.id}/"
            task_issues = [i for i in issues if task_prefix in i or task_slash in i]
            has_missing_data = any(
                kw in i for i in task_issues
                for kw in ("missing", "unscored template", "incomplete", "no valid content", "parse error")
            )

            threshold_ok = True
            if has_qwen and qwen_pr >= 0.7:
                issues.append(f"[{task.id}] qwen passrate {qwen_pr:.4f} >= 0.7")
                threshold_ok = False
            if has_claude and claude_pr < 0.71:
                issues.append(f"[{task.id}] claude passrate {claude_pr:.4f} < 0.71")
                threshold_ok = False
            if has_claude and has_qwen and claude_pr <= qwen_pr:
                issues.append(f"[{task.id}] claude passrate {claude_pr:.4f} <= qwen {qwen_pr:.4f}")
                threshold_ok = False
            if any(not bool(row.get(f"{m}_passrate")) for m in models):
                threshold_ok = False

            if has_missing_data:
                finalize_status = "failed"
            elif not threshold_ok:
                finalize_status = "partial"
            else:
                finalize_status = "done"

            state.set(
                task.id,
                "finalize",
                status=finalize_status,
                qwen_passrate=qwen_pr,
                claude_passrate=claude_pr,
                threshold_ok=threshold_ok,
            )
            rows.append(row)

    csv_path = config.delivery_dir / "submission.csv"
    _write_submission_csv(csv_path, rows)

    print("\nSummary")
    print("=" * 60)
    for row in rows:
        task_id = row["id"]
        task_prefix = f"[{task_id}]"
        task_slash = f"[{task_id}/"
        status = "WARN" if any(task_prefix in issue or task_slash in issue for issue in issues) else "OK"
        print(
            f"  {task_id}: qwen={row.get('qwen_passrate', '-') or '-'} "
            f"claude={row.get('claude_passrate', '-') or '-'} "
            f"[{row.get('qwen_turns', '?')}/{row.get('claude_turns', '?')} turns] {status}"
        )

    if issues:
        print(f"\nIssues ({len(issues)})")
        print("=" * 60)
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\nAll thresholds passed.")

    print(f"\nSubmission CSV: {csv_path}")
    print("Finalize complete.")


def _write_submission_csv(path: Path, rows: list[dict[str, str]]) -> None:
    mapped_rows: list[dict[str, str]] = []
    for row in rows:
        mapped = {"id": row["id"]}
        for csv_col, internal_key in SUBMISSION_KEY_MAP.items():
            mapped[csv_col] = row.get(internal_key, "")
        mapped_rows.append(mapped)

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(mapped_rows)
