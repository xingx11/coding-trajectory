"""check subcommand: deep validation of delivery data quality.

Goes beyond validate.py's file-existence checks to inspect actual content:
trajectory turn counts, model identity, session consistency, score quality.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from ctpipe.config import (
    THRESHOLD_CLAUDE_MIN,
    THRESHOLD_QWEN_MAX,
    THRESHOLD_RELATIVE_GAIN_MIN,
    BatchConfig,
    select_delivery_tasks,
)
from ctpipe.toml_utils import (
    EXPECTED_CRITERIA_COUNT,
    calc_passrate,
    is_complete_rubric,
    is_unscored_template,
    read_quality_toml,
)
from ctpipe.trajectory import find_delivery_trajectory, trajectory_filename

MIN_TURNS = 2
MAX_TURNS = 8
MIN_TRAJECTORY_LINES = 10

QWEN_MODEL_KEYWORDS = ("qwen",)
CLAUDE_MODEL_KEYWORDS = ("claude", "anthropic")


def _count_turns(jsonl_path: Path) -> tuple[int, list[str], str, set[str]]:
    """Parse trajectory JSONL and return (turn_count, models, session_id, issues)."""
    user_turns = 0
    models: list[str] = []
    session_id = ""
    line_count = 0
    issues: set[str] = set()
    models_seen: set[str] = set()

    with jsonl_path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            line_count += 1
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue

            if obj.get("sessionId") and not session_id:
                session_id = obj["sessionId"]

            entry_type = obj.get("type", "")
            if entry_type == "user":
                user_turns += 1

            msg = obj.get("message")
            if isinstance(msg, dict):
                if msg.get("role") == "assistant" and msg.get("model"):
                    model_name = msg["model"]
                    if model_name not in models_seen:
                        models.append(model_name)
                        models_seen.add(model_name)

    if line_count < MIN_TRAJECTORY_LINES:
        issues.add(f"trajectory too short ({line_count} lines)")
    if not session_id:
        issues.add("no session_id found in trajectory")
    if not models:
        issues.add("no model identifiers found in trajectory")

    return user_turns, models, session_id, issues


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
        row = submission_rows.get(task.id)

        for model_name in models:
            prefix = f"[{task.id}/{model_name}]"

            # --- Trajectory checks ---
            traj_path = find_delivery_trajectory(delivery_dir, model_name, task.id)
            if not traj_path:
                issues.append(f"{prefix} trajectory file missing")
                continue

            user_turns, detected_models, session_id, traj_issues = _count_turns(traj_path)

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

            # --- Session ID cross-check ---
            if row and session_id:
                csv_session = row.get(f"{model_name} session id", "")
                if csv_session and csv_session != session_id:
                    issues.append(
                        f"{prefix} session_id mismatch: "
                        f"csv={csv_session!r} vs trajectory={session_id!r}"
                    )

            # --- Score checks ---
            score_path = delivery_dir / "scores" / model_name / f"{task.id}.quality.toml"
            if not score_path.exists():
                issues.append(f"{prefix} score file missing")
                continue

            try:
                criteria = read_quality_toml(score_path)
            except Exception as exc:
                issues.append(f"{prefix} score file parse error: {exc}")
                continue

            if is_unscored_template(criteria):
                issues.append(f"{prefix} score file is still an unscored template")
                continue

            if len(criteria) != EXPECTED_CRITERIA_COUNT:
                issues.append(f"{prefix} wrong criteria count: {len(criteria)} (expected {EXPECTED_CRITERIA_COUNT})")
                continue

            for i, c in enumerate(criteria, 1):
                if c.score < 0 or c.score > 5:
                    issues.append(f"{prefix} criterion {i} ({c.name}): score {c.score} out of range 0-5")
                if not c.rationale:
                    issues.append(f"{prefix} criterion {i} ({c.name}): missing rationale")

            if not is_complete_rubric(criteria):
                scored_n = sum(1 for c in criteria if c.score > 0 or c.rationale)
                issues.append(f"{prefix} incomplete scoring: {scored_n}/{EXPECTED_CRITERIA_COUNT} criteria filled")
                continue

            passrate = calc_passrate(criteria)
            task_stats[f"{model_name}_passrate"] = f"{passrate:.4f}"

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

        # --- Cross-model threshold checks ---
        qwen_pr = float(task_stats.get("qwen_passrate", "0") or "0")
        claude_pr = float(task_stats.get("claude_passrate", "0") or "0")
        has_qwen = "qwen_passrate" in task_stats
        has_claude = "claude_passrate" in task_stats

        if has_qwen and has_claude:
            if qwen_pr >= THRESHOLD_QWEN_MAX:
                issues.append(f"[{task.id}] qwen passrate {qwen_pr:.4f} >= {THRESHOLD_QWEN_MAX}")
            if claude_pr < THRESHOLD_CLAUDE_MIN:
                issues.append(f"[{task.id}] claude passrate {claude_pr:.4f} < {THRESHOLD_CLAUDE_MIN}")
            if claude_pr <= qwen_pr:
                issues.append(f"[{task.id}] claude ({claude_pr:.4f}) not greater than qwen ({qwen_pr:.4f})")
            if qwen_pr > 0:
                relative_gain = (claude_pr - qwen_pr) / qwen_pr
                if relative_gain <= THRESHOLD_RELATIVE_GAIN_MIN:
                    issues.append(
                        f"[{task.id}] relative gain {relative_gain:.2%} <= {THRESHOLD_RELATIVE_GAIN_MIN:.0%}"
                    )
            elif claude_pr < THRESHOLD_RELATIVE_GAIN_MIN:
                issues.append(
                    f"[{task.id}] qwen=0, claude {claude_pr:.4f} too low"
                )

        # --- Metadata check ---
        metadata_path = delivery_dir / "metadata" / f"{task.id}.md"
        if not metadata_path.exists():
            issues.append(f"[{task.id}] metadata file missing")

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
