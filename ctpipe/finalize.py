"""finalize subcommand: calculate passrates and generate submission.csv."""

from __future__ import annotations

import csv
import re
from pathlib import Path

from ctpipe.config import (
    SUBMISSION_FIELDNAMES,
    SUBMISSION_KEY_MAP,
    BatchConfig,
    check_passrate_thresholds,
    select_delivery_tasks,
)
from ctpipe.state import PipelineState
from ctpipe.toml_utils import calc_passrate, is_complete_rubric, is_unscored_template, read_quality_toml
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory, trajectory_filename


def _build_submission_id(person_id: str, delivery_date: str, task_type: str, seq: int) -> str:
    """Build submission ID: {person_id}-{month_day}-{task_type}-{seq:02d}.

    delivery_date is YYYYMMDD, e.g. "20260607" → "607" (month=6, day=07).
    """
    if len(delivery_date) == 8:
        month = str(int(delivery_date[4:6]))  # strip leading zero
        day = delivery_date[6:8]
        date_part = f"{month}{day}"
    else:
        date_part = delivery_date
    return f"{person_id}-{date_part}-{task_type}-{seq:02d}"


def assign_submission_ids(
    tasks: list,
    person_id: str,
    delivery_date: str,
) -> dict[str, str]:
    """Assign submission IDs to tasks, grouped by task_type with sequential numbering."""
    type_counters: dict[str, int] = {}
    id_map: dict[str, str] = {}
    for task in tasks:
        tt = task.task_type
        type_counters[tt] = type_counters.get(tt, 0) + 1
        id_map[task.id] = _build_submission_id(person_id, delivery_date, tt, type_counters[tt])
    return id_map


def finalize(config: BatchConfig, task_ids: list[str] | None = None, models: list[str] | None = None) -> None:
    state = PipelineState(config.state_path)
    # Always compute submission IDs from the full task list so that
    # sequence numbers stay consistent whether or not --tasks is used.
    all_tasks = select_delivery_tasks(config, task_ids=None)
    submission_ids = assign_submission_ids(all_tasks, config.person_id, config.delivery_date)

    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]

    if not tasks:
        print("No tasks found in delivery manifest or tasks.toml; nothing to finalize.")
        return

    rows: list[dict[str, str]] = []
    issues: list[str] = []

    with state.batch():
        for task in tasks:
            row: dict[str, str] = {"id": submission_ids.get(task.id, task.id), "_task_id": task.id}

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
                    if not session_id or not collect_info.get("model_detected"):
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
                    else:
                        detected = collect_info.get("model_detected", "unknown")
                        if detected not in ("unknown", model_name):
                            issues.append(
                                f"[{task.id}/{model_name}] provider mismatch: {detected}"
                            )

                passrate = ""
                if score_path.exists():
                    try:
                        criteria = read_quality_toml(score_path)
                        if is_unscored_template(criteria):
                            issues.append(f"[{task.id}/{model_name}] score file is unscored template")
                        elif not is_complete_rubric(criteria):
                            scored_count = sum(1 for c in criteria if c.score >= 1 and c.rationale)
                            issues.append(
                                f"[{task.id}/{model_name}] score file incomplete: "
                                f"{scored_count}/{len(criteria)} criteria scored"
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

            # Use AI-detected bad pattern from scoring, fall back to task config
            qwen_score_info = state.get(task.id, "score", "qwen")
            ai_bad_pattern = qwen_score_info.get("bad_pattern", "")
            row["bad_pattern"] = ai_bad_pattern or task.bad_pattern

            try:
                qwen_pr = float(row.get("qwen_passrate") or 0)
            except (ValueError, TypeError):
                qwen_pr = 0.0
            try:
                claude_pr = float(row.get("claude_passrate") or 0)
            except (ValueError, TypeError):
                claude_pr = 0.0
            has_qwen = bool(row.get("qwen_passrate"))
            has_claude = bool(row.get("claude_passrate"))

            task_prefix = f"[{task.id}]"
            task_slash = f"[{task.id}/"
            task_issues = [i for i in issues if task_prefix in i or task_slash in i]
            has_missing_data = any(
                kw in i for i in task_issues
                for kw in ("missing", "unscored template", "incomplete", "no valid content", "parse error")
            )

            threshold_issues = check_passrate_thresholds(
                task.id, qwen_pr, claude_pr, has_qwen, has_claude,
            )
            issues.extend(threshold_issues)
            threshold_ok = len(threshold_issues) == 0
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
        sub_id = row["id"]
        internal_id = row.get("_task_id", sub_id)
        task_prefix = f"[{internal_id}]"
        task_slash = f"[{internal_id}/"
        status = "WARN" if any(task_prefix in issue or task_slash in issue for issue in issues) else "OK"
        print(
            f"  {sub_id} ({internal_id}): qwen={row.get('qwen_passrate', '-') or '-'} "
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
    _update_metadata_files(config, tasks, rows, state)
    print("Finalize complete.")


def _update_metadata_files(
    config: BatchConfig,
    tasks: list,
    rows: list[dict[str, str]],
    state: PipelineState,
) -> None:
    """Back-fill session ids, passrates, and round counts into metadata files."""
    row_map = {r.get("_task_id", r["id"]): r for r in rows}
    updated = 0

    for task in tasks:
        metadata_path = config.delivery_dir / "metadata" / f"{task.id}.md"
        if not metadata_path.exists():
            continue

        row = row_map.get(task.id, {})
        content = metadata_path.read_text(encoding="utf-8")
        original = content

        for model in ("qwen", "claude"):
            section_label = "Qwen" if model == "qwen" else "Claude"
            session_id = row.get(f"{model}_session_id", "")
            passrate = row.get(f"{model}_passrate", "")
            run_info = state.get(task.id, "run", model)
            actual_turns = run_info.get("turns", "")

            if session_id:
                content = _fill_field(content, section_label, "Session id", session_id)
            if actual_turns:
                content = _fill_field(content, section_label, "Round count", str(actual_turns))

        qwen_pr = row.get("qwen_passrate", "")
        claude_pr = row.get("claude_passrate", "")
        if qwen_pr:
            content = re.sub(
                r"(- Qwen passrate:)\s*$", rf"\1 {qwen_pr}", content, flags=re.MULTILINE
            )
        if claude_pr:
            content = re.sub(
                r"(- Claude passrate:)\s*$", rf"\1 {claude_pr}", content, flags=re.MULTILINE
            )

        if content != original:
            metadata_path.write_text(content, encoding="utf-8")
            updated += 1

    if updated:
        print(f"\nUpdated {updated} metadata file(s) with session/passrate info.")


def _fill_field(content: str, section: str, field: str, value: str) -> str:
    """Fill an empty metadata field within a specific section.

    Only matches if the field line ends with just whitespace (i.e., is empty).
    Bounded to the section by stopping at the next '##' heading.
    """
    pattern = rf"(## {section} Conversation(?:(?!^## ).)*?- {field}:)\s*$"
    replacement = rf"\1 {value}"
    return re.sub(pattern, replacement, content, count=1, flags=re.MULTILINE | re.DOTALL)


def _write_submission_csv(path: Path, rows: list[dict[str, str]]) -> None:
    mapped_rows: list[dict[str, str]] = []
    for row in rows:
        mapped = {"id": row["id"]}
        for csv_col, internal_key in SUBMISSION_KEY_MAP.items():
            mapped[csv_col] = row.get(internal_key, "")
        mapped_rows.append(mapped)

    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDNAMES)
        writer.writeheader()
        writer.writerows(mapped_rows)
    tmp.replace(path)
