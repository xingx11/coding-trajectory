"""health subcommand: comprehensive pipeline health overview.

Aggregates data from state, stats, check, and validate into a single
at-a-glance health report with a HEALTHY/WARNING/CRITICAL verdict.
"""

from __future__ import annotations

import sys
from pathlib import Path

from ctpipe.config import (
    MAX_TURNS,
    MIN_CRITERIA_COUNT,
    MAX_CRITERIA_COUNT,
    MIN_TRAJECTORY_LINES,
    MIN_TURNS,
    MODEL_SPECIFIC_STAGES,
    THRESHOLD_CLAUDE_MIN,
    THRESHOLD_QWEN_MAX,
    THRESHOLD_RELATIVE_GAIN_MIN,
    BatchConfig,
    check_passrate_thresholds,
    is_valid_criterion_name,
    select_delivery_tasks,
)
from ctpipe.state import PipelineState
from ctpipe.stats import (
    ALL_STAGES,
    _classify,
    _collect_passrate_diff,
    _collect_passrate_stats,
    _collect_per_task,
    _collect_stage_counts,
    _find_bottleneck,
    _find_slowest,
    _fmt_duration,
)
from ctpipe.toml_utils import (
    calc_passrate,
    is_complete_rubric,
    is_unscored_template,
    read_quality_toml,
)
from ctpipe.trajectory import find_delivery_trajectory, parse_trajectory

_MODEL_AGNOSTIC_STAGES = ("prepare", "finalize", "validate")


# ---------------------------------------------------------------------------
# ANSI color helper
# ---------------------------------------------------------------------------

def _ensure_utf8_stdout() -> None:
    """Reconfigure stdout to UTF-8 on Windows (GBK cannot render ✓⚠█ etc.)."""
    if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _color(text: str, code: int) -> str:
    """Wrap *text* in ANSI color codes.  No-op when stdout is not a tty."""
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


# ---------------------------------------------------------------------------
# Blocker / anomaly detection
# ---------------------------------------------------------------------------

def _detect_blockers(
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> dict[str, list]:
    """Find permanently_failed, stuck, and missing-file entries."""
    permanently_failed: list[dict] = []
    stuck: list[dict] = []
    missing_files: list[str] = []

    for tid in task_ids:
        for stage in ALL_STAGES:
            if stage in _MODEL_AGNOSTIC_STAGES:
                info = state.get(tid, stage)
                status = info.get("status", "")
                if status == "permanently_failed":
                    permanently_failed.append(
                        {"task_id": tid, "stage": stage, "model": None}
                    )
                elif status in ("failed", "partial"):
                    retries = info.get("retry_count", 0)
                    if retries >= 2:
                        stuck.append({
                            "task_id": tid,
                            "stage": stage,
                            "model": None,
                            "status": status,
                            "retries": retries,
                        })
            else:
                for model in models:
                    info = state.get(tid, stage, model)
                    status = info.get("status", "")
                    if status == "permanently_failed":
                        permanently_failed.append(
                            {"task_id": tid, "stage": stage, "model": model}
                        )
                    elif status in ("failed", "partial"):
                        retries = info.get("retry_count", 0)
                        if retries >= 2:
                            stuck.append({
                                "task_id": tid,
                                "stage": stage,
                                "model": model,
                                "status": status,
                                "retries": retries,
                            })

    return {
        "permanently_failed": permanently_failed,
        "stuck": stuck,
        "missing_files": missing_files,
    }


# ---------------------------------------------------------------------------
# Threshold violation detection
# ---------------------------------------------------------------------------

def _detect_threshold_violations(
    state: PipelineState,
    task_ids: list[str],
) -> list[str]:
    """Collect passrate threshold violations across all tasks."""
    violations: list[str] = []
    for tid in task_ids:
        info = state.get(tid, "finalize")
        qwen_pr = info.get("qwen_passrate", 0.0)
        claude_pr = info.get("claude_passrate", 0.0)
        has_qwen = isinstance(qwen_pr, (int, float)) and qwen_pr > 0
        has_claude = isinstance(claude_pr, (int, float)) and claude_pr > 0
        qw = float(qwen_pr) if has_qwen else 0.0
        cl = float(claude_pr) if has_claude else 0.0
        violations.extend(check_passrate_thresholds(tid, qw, cl, has_qwen, has_claude))
    return violations


# ---------------------------------------------------------------------------
# Delivery progress
# ---------------------------------------------------------------------------

def _compute_delivery_progress(
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> dict[str, object]:
    """Count tasks where every stage is 'done' for all models."""
    fully_done = 0
    for tid in task_ids:
        all_done = True
        for stage in ALL_STAGES:
            if stage in _MODEL_AGNOSTIC_STAGES:
                if state.get(tid, stage).get("status") != "done":
                    all_done = False
                    break
            else:
                for model in models:
                    if state.get(tid, stage, model).get("status") != "done":
                        all_done = False
                        break
                if not all_done:
                    break
        if all_done:
            fully_done += 1
    total = len(task_ids)
    return {
        "fully_done": fully_done,
        "total": total,
        "pct": round(fully_done / total * 100, 1) if total else 0.0,
    }


# ---------------------------------------------------------------------------
# Health verdict scoring
# ---------------------------------------------------------------------------

def _classify_health_verdict(data: dict) -> tuple[str, int, list[str]]:
    """Compute verdict from health data.

    Returns (verdict_label, total_points, reason_strings).
    """
    points = 0
    reasons: list[str] = []

    # Permanently failed (hard override → CRITICAL)
    perm = data["blockers"]["permanently_failed"]
    if perm:
        pts = len(perm) * 10
        points += pts
        reasons.append(f"{len(perm)} permanently_failed (+{pts})")

    # Failed entries per stage
    for row in data["stages"]:
        failed = int(row["failed"])
        if failed > 0:
            pts = failed * 3
            points += pts
            reasons.append(f"{failed} failed in {row['stage']} (+{pts})")

    # Partial entries per stage
    for row in data["stages"]:
        partial = int(row["partial"])
        if partial > 0:
            pts = partial * 1
            points += pts
            reasons.append(f"{partial} partial in {row['stage']} (+{pts})")

    # Stuck tasks
    stuck = data["blockers"]["stuck"]
    if stuck:
        pts = len(stuck) * 2
        points += pts
        reasons.append(f"{len(stuck)} stuck task(s) (+{pts})")

    # Threshold violations
    violations = data["threshold_violations"]
    if violations:
        pts = len(violations) * 5
        points += pts
        reasons.append(f"{len(violations)} threshold violation(s) (+{pts})")

    # Missing files
    missing = data["blockers"]["missing_files"]
    if missing:
        pts = len(missing) * 4
        points += pts
        reasons.append(f"{len(missing)} missing file(s) (+{pts})")

    # Verdict
    if perm:
        verdict = "CRITICAL"
    elif points >= 15:
        verdict = "CRITICAL"
    elif points >= 1:
        verdict = "WARNING"
    else:
        verdict = "HEALTHY"

    return verdict, points, reasons


# ---------------------------------------------------------------------------
# Progress bar rendering
# ---------------------------------------------------------------------------

def _print_stage_progress_bar(
    stage_name: str, done: int, total: int, width: int = 30,
) -> None:
    """Print one progress bar line."""
    pct = done / total if total else 0
    filled = int(width * pct)
    bar = "█" * filled + "░" * (width - filled)
    print(f"  {stage_name:<15s} [{bar}] {done:>3d}/{total:<3d} {pct:>5.0%}")


# ---------------------------------------------------------------------------
# Per-task table (verbose)
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "done": "✓",      # ✓
    "partial": "◐",   # ◐
    "failed": "✗",    # ✗
    "pending": "·",   # ·
}


def _print_per_task_table(
    per_task: dict[str, dict[str, object]],
    models: list[str],
) -> None:
    """Render per-task detail table for verbose mode."""
    # Build header columns
    cols = ["Task"]
    for stage in ALL_STAGES:
        if stage in _MODEL_AGNOSTIC_STAGES:
            cols.append(stage[:5].title())
        else:
            for m in models:
                cols.append(f"{stage[:3].title()}/{m[0].upper()}")
    for m in models:
        cols.append(f"{m[:1].upper()} pass")

    # Column widths
    widths = [max(len(c), 8) for c in cols]
    widths[0] = max(widths[0], 10)

    header = "  ".join(c.ljust(w) for c, w in zip(cols, widths))
    print(f"  {header}")
    print(f"  {'  '.join('-' * w for w in widths)}")

    for tid in sorted(per_task):
        detail = per_task[tid]
        cells: list[str] = [tid]

        for stage in ALL_STAGES:
            if stage in _MODEL_AGNOSTIC_STAGES:
                st = str(detail.get(stage, "pending"))
                cells.append(_STATUS_ICONS.get(st, st))
            else:
                for m in models:
                    key = f"{stage}/{m}"
                    st = str(detail.get(key, "pending"))
                    cells.append(_STATUS_ICONS.get(st, st))

        for m in models:
            pr = detail.get(f"{m}_passrate")
            if isinstance(pr, (int, float)) and pr > 0:
                cells.append(f"{float(pr):.4f}")
            else:
                cells.append("—")  # —

        row = "  ".join(c.ljust(w) for c, w in zip(cells, widths))
        print(f"  {row}")


# ---------------------------------------------------------------------------
# Main report rendering
# ---------------------------------------------------------------------------

def _print_health_report(data: dict, verbose: bool, models: list[str]) -> None:
    """Render the full terminal health report."""
    verdict = data["verdict"]
    points = data["verdict_points"]

    # Color the verdict
    if verdict == "HEALTHY":
        verdict_str = _color(f"✓ {verdict}", 32)
    elif verdict == "WARNING":
        verdict_str = _color(f"⚠ {verdict}", 33)
    else:
        verdict_str = _color(f"✗ {verdict}", 31)

    print(f"\n{'=' * 60}")
    print(f"  PIPELINE HEALTH — {data['delivery_dir']}")
    print(f"{'=' * 60}")
    print(f"\n  Verdict: {verdict_str}  ({points} points)")

    # -- §1 Delivery Progress --
    prog = data["delivery_progress"]
    print(f"\n{'-' * 60}")
    print(f"  §1  DELIVERY PROGRESS")
    print(f"{'-' * 60}")
    print(f"  Tasks fully done: {prog['fully_done']} / {prog['total']}  ({prog['pct']:.0f}%)")
    pct = prog["pct"] / 100 if prog["total"] else 0
    bar_w = 32
    filled = int(bar_w * pct)
    bar = "█" * filled + "░" * (bar_w - filled)
    print(f"  [{bar}] {prog['pct']:.0f}%")

    # -- §2 Stage Progress --
    print(f"\n{'-' * 60}")
    print(f"  §2  STAGE PROGRESS")
    print(f"{'-' * 60}")
    for row in data["stages"]:
        _print_stage_progress_bar(
            str(row["stage"]), int(row["done"]), int(row["total"]),
        )

    # -- §3 Passrate Summary --
    print(f"\n{'-' * 60}")
    print(f"  §3  PASSRATE SUMMARY")
    print(f"{'-' * 60}")
    pr = data["passrate"]
    if pr:
        hdr = f"  {'Model':<8} {'Min':>8} {'Max':>8} {'Mean':>8} {'Count':>6}  {'Threshold':<10} {'Status'}"
        print(hdr)
        print(f"  {'-' * 66}")
        for model_name, stats in pr.items():
            threshold = stats.get("threshold", 0)
            direction = stats.get("direction", "")
            ok = stats.get("threshold_ok", True)
            status = _color("✓ OK", 32) if ok else _color("✗ FAIL", 31)
            dir_sym = "< " if direction == "below" else "> "
            print(
                f"  {model_name:<8} "
                f"{stats['min']:>8.4f} "
                f"{stats['max']:>8.4f} "
                f"{stats['mean']:>8.4f} "
                f"{stats['count']:>6}  "
                f"{dir_sym}{threshold:<8.3f} "
                f"{status}"
            )

    diff = data.get("passrate_diff")
    if diff:
        rg_ok = data.get("relative_gain_ok", True)
        rg_status = _color("✓ OK", 32) if rg_ok else _color("✗ FAIL", 31)
        print(
            f"\n  Relative gain: mean={diff['mean']:+.3f}, "
            f"median={diff['median']:+.3f}  {rg_status} "
            f"(≥ {THRESHOLD_RELATIVE_GAIN_MIN:.0%})"
        )
        print(
            f"  Passrate diff (claude - qwen): mean={diff['mean']:+.4f}, "
            f"n={diff['count']}"
        )
        print(
            f"    Claude > Qwen: {diff['positive']}   "
            f"Claude ≤ Qwen: {diff['negative']}"
        )
    else:
        print(f"\n  Passrate diff: N/A (fewer than 2 paired tasks)")

    # -- §4 Blockers & Anomalies --
    blockers = data["blockers"]
    violations = data["threshold_violations"]
    print(f"\n{'-' * 60}")
    print(f"  §4  BLOCKERS & ANOMALIES")
    print(f"{'-' * 60}")

    print(f"  Permanently failed: {len(blockers['permanently_failed'])}")
    for entry in blockers["permanently_failed"]:
        model_part = f"/{entry['model']}" if entry["model"] else ""
        print(f"    {_color('✗', 31)} {entry['task_id']}/{entry['stage']}{model_part}")

    print(f"  Stuck tasks:        {len(blockers['stuck'])}")
    for entry in blockers["stuck"]:
        model_part = f"/{entry['model']}" if entry["model"] else ""
        print(
            f"    {_color('✗', 33)} {entry['task_id']}/{entry['stage']}{model_part}: "
            f"{entry['status']} (retry_count={entry['retries']})"
        )

    print(f"  Missing files:      {len(blockers['missing_files'])}")
    for msg in blockers["missing_files"]:
        print(f"    {_color('✗', 33)} {msg}")

    print(f"  Threshold violations: {len(violations)}")
    for v in violations[:10]:
        print(f"    {_color('✗', 31)} {v}")
    if len(violations) > 10:
        print(f"    ... and {len(violations) - 10} more")

    # -- §5 Bottleneck --
    bn = data.get("bottleneck", {})
    slowest = data.get("slowest")
    print(f"\n{'-' * 60}")
    print(f"  §5  BOTTLENECK")
    print(f"{'-' * 60}")
    if bn and bn.get("failed", 0) > 0:
        print(f"  Bottleneck: {bn['stage']} ({bn['failed']} task(s) failed)")
    else:
        print(f"  Bottleneck: none (no failures)")
    if slowest:
        print(
            f"  Slowest:    {slowest['task']}/{slowest['model']} "
            f"({_fmt_duration(slowest['duration_s'])})"
        )
    else:
        print(f"  Slowest:    N/A")

    # -- §6 Per-task detail (verbose only) --
    if verbose and data.get("per_task"):
        print(f"\n{'-' * 60}")
        print(f"  §6  PER-TASK DETAIL")
        print(f"{'-' * 60}")
        _print_per_task_table(data["per_task"], models)

    # -- Closing --
    action_count = (
        len(blockers["stuck"])
        + len(violations)
        + len(blockers["permanently_failed"])
    )
    print(f"\n{'=' * 60}")
    if verdict == "HEALTHY":
        print(f"  HEALTH: {_color('HEALTHY', 32)} — Pipeline is on track")
    else:
        print(f"  HEALTH: {verdict_str} — {action_count} issue(s) need attention")
    print(f"  Run 'ctpipe check' for detailed quality analysis")
    print(f"  Run 'ctpipe retry' to auto-retry failed tasks")
    print(f"{'=' * 60}")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def health(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
    *,
    verbose: bool = False,
    as_json: bool = False,
) -> dict | bool:
    """Compute and display comprehensive pipeline health."""
    models = models or ["qwen", "claude"]
    delivery_dir = config.delivery_dir

    if not delivery_dir.exists():
        msg = (
            f"Delivery directory not found: {delivery_dir}\n"
            f"Run 'ctpipe prepare' first to initialize the pipeline."
        )
        if as_json:
            return {"error": msg, "verdict": "CRITICAL"}
        print(msg)
        return False

    tasks = select_delivery_tasks(config, task_ids)
    if not tasks:
        msg = "No tasks found in delivery manifest or tasks.toml."
        if as_json:
            return {"error": msg, "verdict": "CRITICAL"}
        print(msg)
        return False

    tids = [t.id for t in tasks]
    state = PipelineState(config.state_path)

    # --- Gather data ---
    stage_rows = _collect_stage_counts(state, tids, models)
    passrate_stats = _collect_passrate_stats(state, tids, models)
    passrate_diff = _collect_passrate_diff(state, tids, "qwen", "claude")
    bottleneck = _find_bottleneck(stage_rows)
    blockers = _detect_blockers(state, tids, models)
    violations = _detect_threshold_violations(state, tids)
    progress = _compute_delivery_progress(state, tids, models)

    # Enrich stage rows with pct_done
    for row in stage_rows:
        total = int(row["total"])
        row["pct_done"] = round(int(row["done"]) / total * 100, 1) if total else 0.0

    # Enrich passrate stats with threshold info
    for model_name, stats in passrate_stats.items():
        if model_name == "qwen":
            stats["threshold"] = THRESHOLD_QWEN_MAX
            stats["direction"] = "below"
            stats["threshold_ok"] = stats["mean"] < THRESHOLD_QWEN_MAX
        elif model_name == "claude":
            stats["threshold"] = THRESHOLD_CLAUDE_MIN
            stats["direction"] = "above"
            stats["threshold_ok"] = stats["mean"] > THRESHOLD_CLAUDE_MIN

    # Relative gain check
    relative_gain_ok = True
    if passrate_diff and passrate_diff.get("mean") is not None:
        qwen_mean = passrate_stats.get("qwen", {}).get("mean", 0)
        claude_mean = passrate_stats.get("claude", {}).get("mean", 0)
        if qwen_mean > 0:
            relative_gain_ok = (
                (claude_mean - qwen_mean) / qwen_mean
            ) >= THRESHOLD_RELATIVE_GAIN_MIN
        else:
            relative_gain_ok = claude_mean >= THRESHOLD_RELATIVE_GAIN_MIN

    # Slowest task
    slowest_info = None
    slowest = _find_slowest(state, tids, models)
    if slowest:
        slowest_info = {
            "task": slowest[0],
            "model": slowest[1],
            "duration_s": slowest[2],
        }

    # Per-task (verbose only)
    per_task_data = None
    if verbose:
        per_task_data = _collect_per_task(state, tids, models)

    # --- Assemble health data ---
    data: dict[str, object] = {
        "delivery_date": config.delivery_date,
        "delivery_dir": delivery_dir.name,
        "task_count": len(tasks),
        "stages": stage_rows,
        "passrate": passrate_stats,
        "passrate_diff": passrate_diff,
        "relative_gain_ok": relative_gain_ok,
        "blockers": blockers,
        "threshold_violations": violations,
        "delivery_progress": progress,
        "bottleneck": {"stage": bottleneck[0], "failed": bottleneck[1]},
        "slowest": slowest_info,
        "per_task": per_task_data,
    }

    # --- Compute verdict ---
    verdict, points, reasons = _classify_health_verdict(data)
    data["verdict"] = verdict
    data["verdict_points"] = points
    data["verdict_reasons"] = reasons

    # --- Output ---
    if as_json:
        if not verbose:
            data.pop("per_task", None)
        return data

    _ensure_utf8_stdout()
    _print_health_report(data, verbose, models)
    return verdict == "HEALTHY"


# ---------------------------------------------------------------------------
# Model identity keywords (mirrors check.py, kept local to avoid coupling)
# ---------------------------------------------------------------------------

_QWEN_MODEL_KW = ("qwen",)
_CLAUDE_MODEL_KW = ("claude", "anthropic")


def _model_matches(detected_models: set[str], expected: str) -> bool:
    """Return True if any detected model name hints at *expected* provider."""
    keywords = _QWEN_MODEL_KW if expected == "qwen" else _CLAUDE_MODEL_KW
    return any(
        any(kw in m.lower() for kw in keywords) for m in detected_models
    )


# ---------------------------------------------------------------------------
# health_check — JSON-first structured diagnostic
# ---------------------------------------------------------------------------

def health_check(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
) -> dict:
    """Run comprehensive pipeline health checks and return a JSON-ready dict.

    Top-level fields::

        overall_status       "healthy" | "degraded" | "critical"
        stage_summary        [{stage, done, failed, partial, pending, total, status}, ...]
        threshold_violations [str, ...]
        integrity_issues     [str, ...]

    Status rules:
      - ``critical``  — any ``permanently_failed`` entry exists in pipeline state
      - ``degraded``  — threshold violations or integrity issues are present
      - ``healthy``   — no violations, no integrity issues, no permanent failures
    """
    models = models or ["qwen", "claude"]
    delivery_dir = config.delivery_dir

    # --- early-exit for missing delivery / no tasks ---
    if not delivery_dir.exists():
        return _hc_error_result(
            config, f"delivery directory not found: {delivery_dir}",
        )

    tasks = select_delivery_tasks(config, task_ids)
    if not tasks:
        return _hc_error_result(
            config, "no tasks found in delivery manifest or tasks.toml",
        )

    tids = [t.id for t in tasks]
    state = PipelineState(config.state_path)

    # 1. stage_summary --------------------------------------------------
    stage_summary = _hc_build_stage_summary(state, tids, models)

    # 2. permanently_failed (feeds overall_status) ----------------------
    blockers = _detect_blockers(state, tids, models)
    permanently_failed = blockers["permanently_failed"]

    # 3. threshold_violations -------------------------------------------
    threshold_violations = _hc_collect_threshold_violations(state, tids)

    # 4. integrity_issues (trajectory + scoring) ------------------------
    integrity_issues = _hc_collect_integrity_issues(
        config, delivery_dir, state, tids, models,
    )

    # --- overall_status ------------------------------------------------
    if permanently_failed:
        overall_status = "critical"
    elif threshold_violations or integrity_issues:
        overall_status = "degraded"
    else:
        overall_status = "healthy"

    return {
        "overall_status": overall_status,
        "delivery_date": config.delivery_date,
        "delivery_dir": delivery_dir.name,
        "task_count": len(tasks),
        "permanently_failed": [
            {
                "task_id": e["task_id"],
                "stage": e["stage"],
                **({"model": e["model"]} if e["model"] else {}),
            }
            for e in permanently_failed
        ],
        "stage_summary": stage_summary,
        "threshold_violations": threshold_violations,
        "integrity_issues": integrity_issues,
    }


def _hc_error_result(config: BatchConfig, message: str) -> dict:
    """Build a critical result for early-exit error conditions."""
    return {
        "overall_status": "critical",
        "delivery_date": config.delivery_date,
        "delivery_dir": config.delivery_dir.name,
        "task_count": 0,
        "permanently_failed": [],
        "stage_summary": [],
        "threshold_violations": [],
        "integrity_issues": [message],
    }


# ---------------------------------------------------------------------------
# stage_summary
# ---------------------------------------------------------------------------

def _hc_build_stage_summary(
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> list[dict]:
    """Build per-stage summary rows with a local status field."""
    rows = _collect_stage_counts(state, task_ids, models)
    summary: list[dict] = []
    for row in rows:
        failed = int(row["failed"])
        partial = int(row["partial"])
        pending = int(row["pending"])
        if failed > 0:
            stage_status = "critical"
        elif partial > 0 or pending > 0:
            stage_status = "degraded"
        else:
            stage_status = "healthy"
        summary.append({
            "stage": str(row["stage"]),
            "done": int(row["done"]),
            "failed": failed,
            "partial": partial,
            "pending": pending,
            "total": int(row["total"]),
            "status": stage_status,
        })
    return summary


# ---------------------------------------------------------------------------
# threshold_violations (reuses check_passrate_thresholds)
# ---------------------------------------------------------------------------

def _hc_collect_threshold_violations(
    state: PipelineState,
    task_ids: list[str],
) -> list[str]:
    """Collect passrate threshold violations across all tasks."""
    violations: list[str] = []
    for tid in task_ids:
        info = state.get(tid, "finalize")
        qwen_pr = info.get("qwen_passrate", 0.0)
        claude_pr = info.get("claude_passrate", 0.0)
        has_qwen = isinstance(qwen_pr, (int, float)) and qwen_pr > 0
        has_claude = isinstance(claude_pr, (int, float)) and claude_pr > 0
        if not has_qwen and not has_claude:
            continue
        qw = float(qwen_pr) if has_qwen else 0.0
        cl = float(claude_pr) if has_claude else 0.0
        violations.extend(
            check_passrate_thresholds(tid, qw, cl, has_qwen, has_claude),
        )
    return violations


# ---------------------------------------------------------------------------
# integrity_issues (trajectory + scoring, reuses parse_trajectory /
#                   is_complete_rubric / read_quality_toml)
# ---------------------------------------------------------------------------

def _hc_collect_integrity_issues(
    config: BatchConfig,
    delivery_dir: Path,
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> list[str]:
    """Collect trajectory-integrity and scoring-consistency issues."""
    issues: list[str] = []
    issues.extend(_hc_trajectory_issues(config, delivery_dir, state, task_ids, models))
    issues.extend(_hc_scoring_issues(config, state, task_ids, models))
    return issues


# -- trajectory --

def _hc_trajectory_issues(
    config: BatchConfig,
    delivery_dir: Path,
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> list[str]:
    """Validate trajectory JSONL files via parse_trajectory."""
    issues: list[str] = []

    for tid in task_ids:
        for model in models:
            prefix = f"[{tid}/{model}]"
            traj_path = find_delivery_trajectory(delivery_dir, model, tid)

            if not traj_path:
                run_st = state.get(tid, "run", model).get("status", "")
                col_st = state.get(tid, "collect", model).get("status", "")
                if run_st == "done" or col_st == "done":
                    issues.append(
                        f"{prefix} trajectory file missing "
                        f"(run={run_st}, collect={col_st})"
                    )
                continue

            info = parse_trajectory(traj_path)

            if info.line_count < MIN_TRAJECTORY_LINES:
                issues.append(
                    f"{prefix} trajectory too short "
                    f"({info.line_count} lines, min {MIN_TRAJECTORY_LINES})"
                )
            if not info.session_id:
                issues.append(f"{prefix} no session_id in trajectory")
            if info.user_turns < MIN_TURNS:
                issues.append(
                    f"{prefix} too few user turns: "
                    f"{info.user_turns} (min {MIN_TURNS})"
                )
            if info.models and not _model_matches(info.models, model):
                issues.append(
                    f"{prefix} model mismatch: expected {model}, "
                    f"found {sorted(info.models)[:3]}"
                )

    return issues


# -- scoring --

def _hc_scoring_issues(
    config: BatchConfig,
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> list[str]:
    """Validate score rubric files via read_quality_toml + is_complete_rubric."""
    issues: list[str] = []
    parsed_criteria: dict[str, dict[str, list]] = {}

    for tid in task_ids:
        task_criteria: dict[str, list] = {}
        for model in models:
            prefix = f"[{tid}/{model}]"
            score_path = config.resolve_score_path(tid, model)

            if not score_path.exists():
                ss = state.get(tid, "score", model).get("status", "")
                if ss == "done":
                    issues.append(
                        f"{prefix} score file missing (score status=done)"
                    )
                continue

            try:
                criteria = read_quality_toml(score_path)
            except Exception as exc:
                issues.append(f"{prefix} score file parse error: {exc}")
                continue

            if is_unscored_template(criteria):
                issues.append(f"{prefix} still an unscored template")
                continue

            n = len(criteria)
            if not (MIN_CRITERIA_COUNT <= n <= MAX_CRITERIA_COUNT):
                issues.append(
                    f"{prefix} wrong criteria count: {n} "
                    f"(expected {MIN_CRITERIA_COUNT}-{MAX_CRITERIA_COUNT})"
                )
                continue

            for i, c in enumerate(criteria, 1):
                if not is_valid_criterion_name(c.name):
                    issues.append(
                        f"{prefix} criterion {i}: invalid name {c.name!r}"
                    )
                if c.score < 1 or c.score > 5:
                    issues.append(
                        f"{prefix} criterion {i} ({c.name}): "
                        f"score {c.score} out of range 1-5"
                    )
                if not c.rationale:
                    issues.append(
                        f"{prefix} criterion {i} ({c.name}): missing rationale"
                    )

            if not is_complete_rubric(criteria):
                scored_n = sum(
                    1 for c in criteria if c.score >= 1 and c.rationale
                )
                issues.append(
                    f"{prefix} incomplete rubric: "
                    f"{scored_n}/{n} criteria filled"
                )
                continue

            task_criteria[model] = criteria

        if task_criteria:
            parsed_criteria[tid] = task_criteria

    # cross-model consistency
    for tid, model_criteria in parsed_criteria.items():
        if "qwen" not in model_criteria or "claude" not in model_criteria:
            continue
        qwen_crit = model_criteria["qwen"]
        claude_crit = model_criteria["claude"]

        qwen_names = {c.name for c in qwen_crit}
        claude_names = {c.name for c in claude_crit}
        if qwen_names != claude_names:
            only_q = sorted(qwen_names - claude_names)
            only_c = sorted(claude_names - qwen_names)
            parts: list[str] = []
            if only_q:
                parts.append(f"only in qwen: {', '.join(only_q)}")
            if only_c:
                parts.append(f"only in claude: {', '.join(only_c)}")
            issues.append(
                f"[{tid}] criterion name mismatch between models: "
                f"{'; '.join(parts)}"
            )

        q_desc = {c.name: c.description for c in qwen_crit}
        c_desc = {c.name: c.description for c in claude_crit}
        for name in sorted(set(q_desc) & set(c_desc)):
            if q_desc[name] != c_desc[name]:
                issues.append(
                    f"[{tid}] criterion '{name}' has different descriptions "
                    f"between qwen and claude"
                )

    return issues
