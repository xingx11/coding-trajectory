"""stats subcommand: aggregate pipeline stage statistics.

Shows per-stage status counts (done/partial/failed/pending),
passrate comparison across models, and bottleneck identification.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from ctpipe.config import BatchConfig, select_delivery_tasks
from ctpipe.state import PipelineState

# Stages that do NOT have per-model entries.
_MODEL_AGNOSTIC_STAGES = ("prepare", "finalize", "validate")
# Stages that DO have per-model entries.
_MODEL_SPECIFIC_STAGES = ("run", "collect", "score")
# Ordered pipeline stages.
ALL_STAGES = ("prepare", "run", "collect", "score", "finalize", "validate")
# Status values grouped for display.
_DONE_STATUSES = {"done"}
_PARTIAL_STATUSES = {"partial"}
_FAILED_STATUSES = {"failed"}
_PENDING_STATUSES = {"", "draft"}


def _classify(status: str) -> str:
    """Map a raw status string to one of done/partial/failed/pending."""
    if status in _DONE_STATUSES:
        return "done"
    if status in _PARTIAL_STATUSES:
        return "partial"
    if status in _FAILED_STATUSES:
        return "failed"
    return "pending"


def _collect_stage_counts(
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> list[dict[str, object]]:
    """Count statuses per stage (splitting model-specific stages by model).

    Returns a list of dicts with keys:
        stage, done, partial, failed, pending, total
    """
    rows: list[dict[str, object]] = []

    for stage in ALL_STAGES:
        if stage in _MODEL_AGNOSTIC_STAGES:
            counts = {"done": 0, "partial": 0, "failed": 0, "pending": 0}
            for tid in task_ids:
                info = state.get(tid, stage)
                counts[_classify(info.get("status", ""))] += 1
            rows.append({
                "stage": stage,
                "done": counts["done"],
                "partial": counts["partial"],
                "failed": counts["failed"],
                "pending": counts["pending"],
                "total": len(task_ids),
            })
        else:
            for model in models:
                counts = {"done": 0, "partial": 0, "failed": 0, "pending": 0}
                for tid in task_ids:
                    info = state.get(tid, stage, model)
                    counts[_classify(info.get("status", ""))] += 1
                rows.append({
                    "stage": f"{stage}/{model}",
                    "done": counts["done"],
                    "partial": counts["partial"],
                    "failed": counts["failed"],
                    "pending": counts["pending"],
                    "total": len(task_ids),
                })
    return rows


def _collect_passrate_stats(
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> dict[str, dict[str, float]]:
    """Gather passrate values from finalize state and compute min/max/mean.

    Returns {model: {min, max, mean, count}} for models that have data.
    """
    per_model: dict[str, list[float]] = {m: [] for m in models}
    for tid in task_ids:
        info = state.get(tid, "finalize")
        for model in models:
            key = f"{model}_passrate"
            value = info.get(key)
            if isinstance(value, (int, float)) and value > 0:
                per_model[model].append(float(value))

    result: dict[str, dict[str, float]] = {}
    for model, values in per_model.items():
        if values:
            result[model] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "count": len(values),
            }
    return result


def _collect_passrate_diff(
    state: PipelineState,
    task_ids: list[str],
    model_a: str,
    model_b: str,
) -> dict[str, float] | None:
    """Compute per-task passrate difference (model_b - model_a) and return stats.

    Only includes tasks where both models have a passrate.
    Returns {mean, median, std, count, positive, negative} or None when
    fewer than two tasks have paired data.
    """
    diffs: list[float] = []
    for tid in task_ids:
        info = state.get(tid, "finalize")
        a_val = info.get(f"{model_a}_passrate")
        b_val = info.get(f"{model_b}_passrate")
        if (
            isinstance(a_val, (int, float)) and a_val > 0
            and isinstance(b_val, (int, float)) and b_val > 0
        ):
            diffs.append(float(b_val) - float(a_val))

    if len(diffs) < 2:
        if len(diffs) == 1:
            d = diffs[0]
            return {
                "mean": d,
                "median": d,
                "std": 0.0,
                "count": 1,
                "positive": int(d > 0),
                "negative": int(d < 0),
            }
        return None

    positive = sum(1 for d in diffs if d > 0)
    negative = sum(1 for d in diffs if d < 0)
    return {
        "mean": statistics.mean(diffs),
        "median": statistics.median(diffs),
        "std": statistics.stdev(diffs),
        "count": len(diffs),
        "positive": positive,
        "negative": negative,
    }


def _find_bottleneck(rows: list[dict[str, object]]) -> tuple[str, int]:
    """Return (stage_name, failed_count) for the stage with the most failures.

    Ties are broken by partial + pending count (more work remaining wins).
    """
    worst_stage = ""
    worst_failed = 0
    worst_remaining = 0
    for row in rows:
        failed = int(row["failed"])  # type: ignore[arg-type]
        remaining = int(row["partial"]) + int(row["pending"])  # type: ignore[arg-type]
        if failed > worst_failed or (failed == worst_failed and failed > 0 and remaining > worst_remaining):
            worst_failed = failed
            worst_remaining = remaining
            worst_stage = str(row["stage"])
    return worst_stage, worst_failed


def _print_table(
    stage_rows: list[dict[str, object]],
    passrate_stats: dict[str, dict[str, float]],
    passrate_diff: dict[str, float] | None,
    bottleneck: tuple[str, int],
    model_a: str,
    model_b: str,
) -> None:
    """Render stats as a human-readable table."""
    print("\n" + "=" * 70)
    print("PIPELINE STATS")
    print("=" * 70)

    # Stage counts table
    header = f"{'Stage':<18} {'Done':>5} {'Part':>5} {'Fail':>5} {'Pend':>5} {'Total':>6}"
    print(f"\n{header}")
    print("-" * len(header))
    for row in stage_rows:
        print(
            f"{str(row['stage']):<18} "
            f"{int(row['done']):>5} "
            f"{int(row['partial']):>5} "
            f"{int(row['failed']):>5} "
            f"{int(row['pending']):>5} "
            f"{int(row['total']):>6}"
        )

    # Passrate summary per model
    if passrate_stats:
        print(f"\n{'Model':<10} {'Min':>8} {'Max':>8} {'Mean':>8} {'Count':>6}")
        print("-" * 42)
        for model, s in passrate_stats.items():
            print(
                f"{model:<10} "
                f"{s['min']:>8.4f} "
                f"{s['max']:>8.4f} "
                f"{s['mean']:>8.4f} "
                f"{int(s['count']):>6}"
            )

    # Passrate difference (model_b - model_a)
    if passrate_diff:
        label = f"{model_b} - {model_a}"
        print(f"\nPassrate diff ({label}), n={int(passrate_diff['count'])}")
        print("-" * 46)
        print(f"  Mean    {passrate_diff['mean']:>+.4f}")
        print(f"  Median  {passrate_diff['median']:>+.4f}")
        print(f"  Std     {passrate_diff['std']:> .4f}")
        print(f"  {model_b}>{model_a}: {int(passrate_diff['positive'])}   "
              f"{model_b}<={model_a}: {int(passrate_diff['negative'])}")

    # Bottleneck (stage with most failures)
    stage_name, failed_count = bottleneck
    if failed_count > 0:
        print(f"\nBottleneck: {stage_name} ({failed_count} task(s) failed)")
    else:
        print("\nNo failures. All stages complete or pending.")

    print("=" * 70)


def _collect_per_task(
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> dict[str, dict[str, str]]:
    """Return per-task status breakdown.

    Returns {task_id: {stage_key: classified_status, ...}, ...}.
    For model-agnostic stages the key is the stage name (e.g. "prepare").
    For model-specific stages the key is "stage/model" (e.g. "run/qwen").
    """
    result: dict[str, dict[str, str]] = {}
    for tid in task_ids:
        task_detail: dict[str, str] = {}
        for stage in ALL_STAGES:
            if stage in _MODEL_AGNOSTIC_STAGES:
                info = state.get(tid, stage)
                task_detail[stage] = _classify(info.get("status", ""))
            else:
                for model in models:
                    info = state.get(tid, stage, model)
                    task_detail[f"{stage}/{model}"] = _classify(info.get("status", ""))
        # Attach passrate values when available.
        fin = state.get(tid, "finalize")
        for model in models:
            pr = fin.get(f"{model}_passrate")
            if isinstance(pr, (int, float)) and pr > 0:
                task_detail[f"{model}_passrate"] = round(float(pr), 4)
        result[tid] = task_detail
    return result


def _print_json(
    stage_rows: list[dict[str, object]],
    passrate_stats: dict[str, dict[str, float]],
    passrate_diff: dict[str, float] | None,
    bottleneck: tuple[str, int],
    per_task: dict[str, dict[str, str]],
    model_a: str,
    model_b: str,
) -> None:
    """Render stats as structured JSON with summary and per_task sections."""
    summary: dict[str, object] = {
        "stages": [
            {
                "stage": row["stage"],
                "done": row["done"],
                "partial": row["partial"],
                "failed": row["failed"],
                "pending": row["pending"],
                "total": row["total"],
            }
            for row in stage_rows
        ],
        "passrates": {
            model: {
                "min": round(s["min"], 4),
                "max": round(s["max"], 4),
                "mean": round(s["mean"], 4),
                "count": int(s["count"]),
            }
            for model, s in passrate_stats.items()
        },
        "passrate_diff": None,
        "bottleneck": {
            "stage": bottleneck[0],
            "failed": bottleneck[1],
        },
    }
    if passrate_diff is not None:
        summary["passrate_diff"] = {
            "model_a": model_a,
            "model_b": model_b,
            "mean": round(passrate_diff["mean"], 4),
            "median": round(passrate_diff["median"], 4),
            "std": round(passrate_diff["std"], 4),
            "count": int(passrate_diff["count"]),
            "positive": int(passrate_diff["positive"]),
            "negative": int(passrate_diff["negative"]),
        }

    payload = {"summary": summary, "per_task": per_task}
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def show_stats(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
    fmt: str = "table",
) -> bool:
    """Print aggregate pipeline statistics and return True if all done."""
    models = models or ["qwen", "claude"]

    delivery_dir = config.delivery_dir
    if not delivery_dir.exists():
        msg = f"Delivery directory not found: {delivery_dir}\nRun 'ctpipe prepare' first to initialize the pipeline."
        if fmt == "json":
            print(json.dumps({"error": msg}, indent=2, ensure_ascii=False))
        else:
            print(msg)
        return False

    tasks = select_delivery_tasks(config, task_ids)
    state = PipelineState(delivery_dir / "pipeline_state.json")

    if not tasks:
        print("No tasks found in delivery manifest or tasks.toml.")
        return False

    tids = [t.id for t in tasks]

    stage_rows = _collect_stage_counts(state, tids, models)
    passrate_stats = _collect_passrate_stats(state, tids, models)
    bottleneck = _find_bottleneck(stage_rows)

    # Passrate diff is meaningful when at least two models are compared.
    model_a = models[0] if models else "qwen"
    model_b = models[1] if len(models) >= 2 else "claude"
    passrate_diff = _collect_passrate_diff(state, tids, model_a, model_b) if model_a != model_b else None

    if fmt == "json":
        per_task = _collect_per_task(state, tids, models)
        _print_json(stage_rows, passrate_stats, passrate_diff, bottleneck, per_task, model_a, model_b)
    else:
        _print_table(stage_rows, passrate_stats, passrate_diff, bottleneck, model_a, model_b)

    # Return True only when no failures and nothing pending.
    all_ok = all(
        int(row["failed"]) == 0 and int(row["pending"]) == 0  # type: ignore[arg-type]
        for row in stage_rows
    )
    return all_ok
