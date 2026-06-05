"""finalize subcommand: calculate passrates and generate submission.csv."""

from __future__ import annotations

import csv
from pathlib import Path

from ctpipe.config import BatchConfig, select_tasks
from ctpipe.state import PipelineState
from ctpipe.toml_utils import calc_passrate, read_quality_toml
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory, trajectory_filename


def finalize(config: BatchConfig, task_ids: list[str] | None = None) -> None:
    state = PipelineState(config.delivery_dir / "pipeline_state.json")
    tasks = select_tasks(config.tasks, task_ids)

    rows: list[dict[str, str]] = []
    issues: list[str] = []

    for task in tasks:
        row: dict[str, str] = {"id": task.id}

        for model_name in ("qwen", "claude"):
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
                    if any(item.score > 0 for item in criteria):
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

        qwen_pr = float(row["qwen_passrate"] or 0)
        claude_pr = float(row["claude_passrate"] or 0)
        has_qwen = bool(row["qwen_passrate"])
        has_claude = bool(row["claude_passrate"])

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

        state.set(
            task.id,
            "finalize",
            status="done",
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
        status = "WARN" if any(task_id in issue for issue in issues) else "OK"
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
    fieldnames = [
        "id",
        "qwen 本地trajectory",
        "qwen session id",
        "qwen rubrics 人工评分",
        "claude 本地trajectory",
        "claude session id",
        "claude rubrics 人工评分",
        "qwen passrate",
        "claude passrate",
        "任务类型",
        "应用领域",
        "编程语言",
    ]

    mapped_rows: list[dict[str, str]] = []
    for row in rows:
        mapped_rows.append({
            "id": row["id"],
            "qwen 本地trajectory": row.get("qwen_trajectory", ""),
            "qwen session id": row.get("qwen_session_id", ""),
            "qwen rubrics 人工评分": row.get("qwen_score_path", ""),
            "claude 本地trajectory": row.get("claude_trajectory", ""),
            "claude session id": row.get("claude_session_id", ""),
            "claude rubrics 人工评分": row.get("claude_score_path", ""),
            "qwen passrate": row.get("qwen_passrate", ""),
            "claude passrate": row.get("claude_passrate", ""),
            "任务类型": row.get("task_type", ""),
            "应用领域": row.get("domain", ""),
            "编程语言": row.get("language", ""),
        })

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(mapped_rows)
