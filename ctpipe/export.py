"""export subcommand: export delivery results as a structured JSON report.

Produces a JSON document with three top-level fields:
  - batch_info: delivery metadata (date, person, task count)
  - tasks: per-task array with metadata, trajectory, scoring, threshold data
  - summary: batch-level aggregates

Missing data fields are set to null rather than raising errors.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

from ctpipe.config import (
    BatchConfig,
    TaskConfig,
    check_passrate_thresholds,
    select_delivery_tasks,
)
from ctpipe.state import PipelineState
from ctpipe.toml_utils import Criterion, calc_passrate, read_quality_toml
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory


def _is_under(path: Path, base: Path) -> bool:
    """Check if path is under base directory."""
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


def _safe_trajectory_info(
    delivery_dir: Path,
    model_name: str,
    task_id: str,
    state: PipelineState,
) -> dict[str, Any] | None:
    """Build trajectory_info for one model. Returns None when no data exists."""
    run_info = state.get(task_id, "run", model_name)
    collect_info = state.get(task_id, "collect", model_name)

    session_id = collect_info.get("session_id", run_info.get("session_id", ""))
    turns = run_info.get("turns")
    duration_s = run_info.get("duration_s")

    traj_path = find_delivery_trajectory(
        delivery_dir,
        model_name,
        task_id,
        session_id=session_id or None,
    )

    # Try enriching from trajectory file when state data is sparse.
    if traj_path and traj_path.exists():
        try:
            traj_info = parse_trajectory(traj_path)
            if not session_id:
                session_id = traj_info.session_id
            if turns is None:
                turns = traj_info.user_turns or None
        except Exception as exc:
            print(f"  WARNING: could not parse trajectory {traj_path.name}: {exc}")

    if not session_id and turns is None and duration_s is None and traj_path is None:
        return None

    return {
        "session_id": session_id or None,
        "turns": int(turns) if isinstance(turns, (int, float)) else None,
        "duration_s": round(float(duration_s), 1) if isinstance(duration_s, (int, float)) else None,
    }


def _safe_scoring(
    delivery_dir: Path,
    model_name: str,
    task_id: str,
) -> dict[str, Any] | None:
    """Build scoring block for one model. Returns None when score file missing."""
    score_path = delivery_dir / "scores" / model_name / f"{task_id}.quality.toml"
    if not score_path.exists():
        return None

    try:
        criteria = read_quality_toml(score_path)
    except Exception as exc:
        print(f"  WARNING: could not read score file: {exc}")
        return None

    if not criteria:
        return None

    criteria_list: list[dict[str, Any]] = []
    for c in criteria:
        criteria_list.append({
            "name": c.name,
            "score": c.score if c.score >= 1 else None,
            "weight": c.weight,
            "rationale": c.rationale or None,
        })

    passrate: float | None = None
    try:
        passrate = round(calc_passrate(criteria), 4)
    except (ValueError, ZeroDivisionError):
        pass

    return {
        "criteria": criteria_list,
        "passrate": passrate,
    }


def _build_threshold_check(
    task_id: str,
    qwen_passrate: float | None,
    claude_passrate: float | None,
) -> dict[str, Any]:
    """Build threshold_check block using config.check_passrate_thresholds."""
    qw = qwen_passrate if qwen_passrate is not None else 0.0
    cl = claude_passrate if claude_passrate is not None else 0.0
    has_qwen = qwen_passrate is not None and qwen_passrate > 0
    has_claude = claude_passrate is not None and claude_passrate > 0

    issues = check_passrate_thresholds(task_id, qw, cl, has_qwen, has_claude)
    return {
        "passed": len(issues) == 0 and (has_qwen or has_claude),
        "issues": issues if issues else None,
    }


def _build_task_entry(
    task: TaskConfig,
    delivery_dir: Path,
    state: PipelineState,
    models: list[str],
) -> dict[str, Any]:
    """Build one task entry for the report."""
    metadata: dict[str, Any] = {
        "id": task.id,
        "task_type": task.task_type or None,
        "domain": task.domain or None,
        "language": task.language or None,
        "bad_pattern": task.bad_pattern or None,
    }

    trajectory_info: dict[str, Any] = {}
    scoring_info: dict[str, Any] = {}
    passrates: dict[str, float | None] = {}

    for model_name in models:
        trajectory_info[model_name] = _safe_trajectory_info(
            delivery_dir, model_name, task.id, state,
        )
        scoring_info[model_name] = _safe_scoring(delivery_dir, model_name, task.id)
        sc = scoring_info[model_name]
        passrates[model_name] = sc["passrate"] if sc else None

    threshold_check = _build_threshold_check(
        task.id,
        passrates.get("qwen"),
        passrates.get("claude"),
    )

    return {
        "metadata": metadata,
        "trajectory_info": trajectory_info,
        "scoring": scoring_info,
        "threshold_check": threshold_check,
    }


def _build_summary(
    task_entries: list[dict[str, Any]],
    models: list[str],
) -> dict[str, Any]:
    """Build the summary block from task entries."""
    total = len(task_entries)
    threshold_passed = sum(
        1 for t in task_entries if t["threshold_check"].get("passed")
    )

    per_model: dict[str, dict[str, Any]] = {}
    for model_name in models:
        values: list[float] = []
        for t in task_entries:
            sc = t["scoring"].get(model_name)
            if sc and isinstance(sc.get("passrate"), (int, float)):
                values.append(sc["passrate"])

        if values:
            per_model[model_name] = {
                "min": round(min(values), 4),
                "max": round(max(values), 4),
                "mean": round(statistics.mean(values), 4),
                "count": len(values),
            }
        else:
            per_model[model_name] = None

    return {
        "total_tasks": total,
        "threshold_passed": threshold_passed,
        "per_model_passrate": per_model,
    }


def export_report(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
    output: Path | None = None,
) -> dict[str, Any]:
    """Build and write the JSON export report. Returns the report dict."""
    models = models or ["qwen", "claude"]
    delivery_dir = config.delivery_dir
    state = PipelineState(config.state_path)
    tasks = select_delivery_tasks(config, task_ids)

    # -- batch_info --
    batch_info: dict[str, Any] = {
        "delivery_date": config.delivery_date or None,
        "person_id": config.person_id or None,
        "task_count": len(tasks),
    }

    # -- tasks --
    task_entries: list[dict[str, Any]] = []
    for task in tasks:
        entry = _build_task_entry(task, delivery_dir, state, models)
        task_entries.append(entry)

    # -- summary --
    summary = _build_summary(task_entries, models)

    report: dict[str, Any] = {
        "batch_info": batch_info,
        "tasks": task_entries,
        "summary": summary,
    }

    json_text = json.dumps(report, indent=2, ensure_ascii=False)

    if output:
        # Validate output path is under delivery_dir or base_dir
        resolved = output.resolve()
        allowed = [delivery_dir.resolve(), config.base_dir.resolve()]
        if not any(_is_under(resolved, base) for base in allowed):
            print(f"WARNING: output path {output} is outside project directories, writing anyway")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json_text, encoding="utf-8")
        print(f"Report exported to {output} ({len(tasks)} task(s))")
    else:
        print(json_text)

    return report
