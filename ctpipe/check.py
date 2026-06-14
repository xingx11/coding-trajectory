"""check subcommand: deep validation of delivery data quality.

Goes beyond validate.py's file-existence checks to inspect actual content:
trajectory turn counts, model identity, session consistency, score quality.
"""

from __future__ import annotations

import csv
from pathlib import Path

from ctpipe.config import (
    MAX_TURNS,
    MIN_TRAJECTORY_LINES,
    MIN_TURNS,
    is_valid_criterion_name,
    BatchConfig,
    check_passrate_thresholds,
    select_delivery_tasks,
)
from ctpipe.finalize import assign_submission_ids
from ctpipe.toml_utils import read_quality_toml, safe_calc_passrate
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory, trajectory_filename

QWEN_MODEL_KEYWORDS = ("qwen",)
CLAUDE_MODEL_KEYWORDS = ("claude", "anthropic")

# Keywords indicating negative evidence in rationale (used in score-rationale contradiction check)
_NEG_KEYWORDS = ("没有", "没看到", "未完成", "不够", "缺少", "缺乏", "不足",
                 "欠缺", "偏弱", "不完整", "还差", "没做", "不到位", "不算完成")

# Keywords indicating strong positive evidence in rationale (used in low-score contradiction check)
_POS_KEYWORDS = ("完整实现", "全部完成", "准确定位", "正确修复", "覆盖全面",
                 "高质量", "完美", "出色", "处理得当", "完整覆盖",
                 "成功修复", "成功实现", "精准定位", "全面覆盖")


def _count_turns(jsonl_path: Path) -> tuple[int, list[str], str, str, set[str]]:
    """Parse trajectory JSONL and return (turn_count, models, session_id, first_user_query, issues)."""
    info = parse_trajectory(jsonl_path)
    issues: set[str] = set()
    if info.line_count < MIN_TRAJECTORY_LINES:
        issues.add(f"trajectory too short ({info.line_count} lines)")
    if not info.session_id:
        issues.add("no session_id found in trajectory")
    if not info.models:
        issues.add("no model identifiers found in trajectory")
    return info.user_turns, sorted(info.models), info.session_id, info.first_user_query, issues


def _check_model_identity(models: list[str], expected: str) -> str | None:
    """Check if detected models match the expected provider."""
    if not models:
        return "no models detected"

    if expected == "qwen":
        if not any(any(kw in m.lower() for kw in QWEN_MODEL_KEYWORDS) for m in models):
            return f"expected qwen model, found: {models}"
    elif expected == "claude":
        if not any(any(kw in m.lower() for kw in CLAUDE_MODEL_KEYWORDS) for m in models):
            return f"expected claude model, found: {models}"
    return None


def check(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
) -> bool:
    """Run deep quality checks on the delivery batch."""
    # Always compute submission IDs from the full task list so that
    # sequence numbers stay consistent whether or not --tasks is used.
    all_tasks = select_delivery_tasks(config, task_ids=None)
    submission_id_map = assign_submission_ids(all_tasks, config.person_id, config.delivery_date)

    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]
    delivery_dir = config.delivery_dir
    issues: list[str] = []
    warnings: list[str] = []
    stats: dict[str, dict[str, str]] = {}

    if not tasks:
        issues.append("no tasks found in delivery manifest or tasks.toml")
        _print_report(issues, warnings, stats)
        return False

    submission_rows: dict[str, dict[str, str]] = {}
    submission_path = delivery_dir / "submission.csv"
    if submission_path.exists():
        with submission_path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if row.get("id"):
                    submission_rows[row["id"]] = row
    else:
        issues.append("submission.csv missing")

    for task in tasks:
        task_stats: dict[str, str] = {}
        parsed_criteria: dict[str, list] = {}  # model -> list[Criterion]
        first_queries: dict[str, str] = {}  # model -> first user query text
        sub_id = submission_id_map.get(task.id, task.id)
        row = submission_rows.get(sub_id)

        for model_name in models:
            prefix = f"[{task.id}/{model_name}]"

            # --- Trajectory checks ---
            traj_path = find_delivery_trajectory(delivery_dir, model_name, task.id)
            if not traj_path:
                issues.append(f"{prefix} trajectory file missing")
                continue

            user_turns, detected_models, session_id, first_query, traj_issues = _count_turns(traj_path)

            for ti in traj_issues:
                issues.append(f"{prefix} {ti}")

            if user_turns < MIN_TURNS:
                issues.append(f"{prefix} too few turns: {user_turns} (min {MIN_TURNS})")
            elif user_turns > MAX_TURNS:
                warnings.append(f"{prefix} high turn count: {user_turns} (max expected {MAX_TURNS})")

            model_issue = _check_model_identity(detected_models, model_name)
            if model_issue:
                issues.append(f"{prefix} model mismatch: {model_issue}")

            task_stats[f"{model_name}_turns"] = str(user_turns)
            task_stats[f"{model_name}_models"] = ",".join(detected_models[:3])
            task_stats[f"{model_name}_session"] = session_id[:12] + "..." if len(session_id) > 12 else session_id
            if first_query:
                first_queries[model_name] = first_query

            # --- Session ID cross-check ---
            if row and session_id:
                csv_session = row.get(f"{model_name} session id", "")
                if csv_session and csv_session != session_id:
                    issues.append(
                        f"{prefix} session_id mismatch: "
                        f"csv={csv_session!r} vs trajectory={session_id!r}"
                    )

            # --- Score checks ---
            score_path = config.resolve_score_path(task.id, model_name)
            passrate, score_err = safe_calc_passrate(score_path)
            if passrate is None:
                issues.append(f"{prefix} {score_err}")
                continue

            criteria = read_quality_toml(score_path)
            for i, c in enumerate(criteria, 1):
                if not is_valid_criterion_name(c.name):
                    issues.append(f"{prefix} criterion {i}: invalid name {c.name!r}")

            task_stats[f"{model_name}_passrate"] = f"{passrate:.4f}"
            task_stats[f"{model_name}_criteria"] = ",".join(c.name for c in criteria)
            parsed_criteria[model_name] = criteria

            # --- Score-rationale contradiction check ---
            for ci, c in enumerate(criteria, 1):
                if c.score >= 4 and c.rationale:
                    neg_hits = [kw for kw in _NEG_KEYWORDS if kw in c.rationale]
                    if neg_hits:
                        warnings.append(
                            f"[{task.id}/{model_name}] criterion {ci} ({c.name}): "
                            f"score={c.score} but rationale contains negative keywords "
                            f"{neg_hits}"
                        )
                if c.score <= 2 and c.rationale:
                    pos_hits = [kw for kw in _POS_KEYWORDS if kw in c.rationale]
                    if pos_hits:
                        warnings.append(
                            f"[{task.id}/{model_name}] criterion {ci} ({c.name}): "
                            f"score={c.score} but rationale contains positive keywords "
                            f"{pos_hits}"
                        )

            if row:
                csv_pr = row.get(f"{model_name} passrate", "")
                if csv_pr:
                    try:
                        csv_pr_val = float(csv_pr)
                    except ValueError:
                        issues.append(f"{prefix} invalid passrate in CSV: {csv_pr!r}")
                    else:
                        if abs(csv_pr_val - passrate) > 0.0001:
                            issues.append(
                                f"{prefix} passrate mismatch: csv={csv_pr} vs computed={passrate:.4f}"
                            )

        # --- B1-d: Cross-model first-query consistency check ---
        if len(first_queries) == 2:
            q_query = first_queries.get("qwen", "")
            c_query = first_queries.get("claude", "")
            if q_query and c_query and q_query != c_query:
                issues.append(
                    f"[{task.id}] B1-d: first user query differs between qwen and claude "
                    f"(qwen={q_query[:60]!r}... vs claude={c_query[:60]!r}...)"
                )

        # --- Cross-model paired consistency check ---
        qwen_names = set(filter(None, task_stats.get("qwen_criteria", "").split(",")))
        claude_names = set(filter(None, task_stats.get("claude_criteria", "").split(",")))
        if qwen_names and claude_names and qwen_names != claude_names:
            only_qwen = qwen_names - claude_names
            only_claude = claude_names - qwen_names
            diff_parts: list[str] = []
            if only_qwen:
                diff_parts.append(f"only in qwen: {', '.join(sorted(only_qwen))}")
            if only_claude:
                diff_parts.append(f"only in claude: {', '.join(sorted(only_claude))}")
            issues.append(f"[{task.id}] criterion name mismatch between qwen/claude: {'; '.join(diff_parts)}")

        # --- Cross-model description consistency check ---
        qwen_crit = parsed_criteria.get("qwen", [])
        claude_crit = parsed_criteria.get("claude", [])
        if qwen_crit and claude_crit:
            q_desc = {c.name: c.description for c in qwen_crit}
            c_desc = {c.name: c.description for c in claude_crit}
            shared_names = set(q_desc) & set(c_desc)
            for name in sorted(shared_names):
                if q_desc[name] != c_desc[name]:
                    issues.append(f"[{task.id}] criterion '{name}' has different descriptions between qwen and claude")

        # --- Cross-model trajectory turn parity check ---
        qwen_turns_str = task_stats.get("qwen_turns", "")
        claude_turns_str = task_stats.get("claude_turns", "")
        if qwen_turns_str and claude_turns_str:
            qwen_turns = int(qwen_turns_str)
            claude_turns = int(claude_turns_str)
            turn_diff = abs(qwen_turns - claude_turns)
            if turn_diff >= 2:
                fewer = "qwen" if qwen_turns < claude_turns else "claude"
                warnings.append(
                    f"[{task.id}] trajectory turn imbalance: "
                    f"qwen={qwen_turns}, claude={claude_turns} "
                    f"(diff={turn_diff}, {fewer} may be truncated — "
                    f"scoring fairness at risk)"
                )

        # --- Cross-model threshold checks ---
        qwen_pr = float(task_stats.get("qwen_passrate", "0") or "0")
        claude_pr = float(task_stats.get("claude_passrate", "0") or "0")
        has_qwen = "qwen_passrate" in task_stats
        has_claude = "claude_passrate" in task_stats

        issues.extend(check_passrate_thresholds(
            task.id, qwen_pr, claude_pr, has_qwen, has_claude,
        ))

        # --- Metadata check ---
        metadata_path = delivery_dir / "metadata" / f"{task.id}.md"
        if not metadata_path.exists():
            issues.append(f"[{task.id}] metadata file missing")

        # --- Prompt/followup parity check ---
        q_followups = getattr(task, "followups_qwen", None) or []
        c_followups = getattr(task, "followups_claude", None) or []
        if len(q_followups) != len(c_followups):
            warnings.append(
                f"[{task.id}] followup count mismatch: "
                f"qwen={len(q_followups)}, claude={len(c_followups)} "
                f"(prompt fairness at risk)"
            )

        stats[task.id] = task_stats

    _print_report(issues, warnings, stats)
    return len(issues) == 0


def _print_report(
    issues: list[str],
    warnings: list[str],
    stats: dict[str, dict[str, str]],
) -> None:
    print("\n" + "=" * 70)
    print("DEEP CHECK REPORT")
    print("=" * 70)

    if stats:
        print(f"\n{'Task':<10} {'Q turns':<8} {'C turns':<8} {'Q pass':<8} {'C pass':<8} {'Q model':<20} {'C model':<20}")
        print("-" * 70)
        for task_id, s in stats.items():
            print(
                f"{task_id:<10} "
                f"{s.get('qwen_turns', '-'):<8} "
                f"{s.get('claude_turns', '-'):<8} "
                f"{s.get('qwen_passrate', '-'):<8} "
                f"{s.get('claude_passrate', '-'):<8} "
                f"{s.get('qwen_models', '-'):<20} "
                f"{s.get('claude_models', '-'):<20}"
            )

    if warnings:
        print(f"\nWarnings ({len(warnings)}):")
        for w in warnings:
            print(f"  * {w}")

    if issues:
        print(f"\nFAILED - {len(issues)} issue(s):")
        for i in issues:
            print(f"  ✗ {i}")
    else:
        print("\nPASSED - All checks OK")

    print("=" * 70)
