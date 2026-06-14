"""stats subcommand: aggregate pipeline stage statistics.

Shows per-stage status counts (done/partial/failed/pending),
passrate comparison across models, and bottleneck identification.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Callable

from ctpipe.config import (
    THRESHOLD_CLAUDE_MIN,
    THRESHOLD_QWEN_MAX,
    THRESHOLD_RELATIVE_GAIN_MIN,
    BatchConfig,
    select_delivery_tasks,
)
from ctpipe.state import PipelineState

# Stages that do NOT have per-model entries.
_MODEL_AGNOSTIC_STAGES = ("prepare", "finalize", "validate")
# Ordered pipeline stages.
ALL_STAGES = ("prepare", "run", "collect", "score", "finalize", "validate")
# Status values grouped for display.
_DONE_STATUSES = {"done"}
_PARTIAL_STATUSES = {"partial"}
_FAILED_STATUSES = {"failed"}
_PENDING_STATUSES = {"", "draft"}

# Whitelist of field names allowed in --filter expressions.
# Restricting to known TaskConfig / finalize-state fields prevents
# expression-injection through arbitrary dict look-ups.
_FILTER_FIELD_WHITELIST = frozenset({
    "task_type",
    "domain",
    "language",
    "bad_pattern",
    "qwen_passrate",
    "claude_passrate",
    "relative_gain",
    "threshold_ok",
    "finalize_status",
    # 交付红线指标（逐条对应 check_passrate_thresholds 的 4 项检查）
    "qwen_over_threshold",
    "claude_under_threshold",
    "claude_not_better",
    "gain_below_threshold",
})


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
    """Gather passrate values from finalize state and compute distribution stats.

    Returns {model: {min, max, mean, median, std, count}} for models that have data.
    """
    per_model: dict[str, list[float]] = {m: [] for m in models}
    for tid in task_ids:
        info = state.get(tid, "finalize")
        for model in models:
            key = f"{model}_passrate"
            value = info.get(key)
            if isinstance(value, (int, float)):
                per_model[model].append(float(value))

    result: dict[str, dict[str, float]] = {}
    for model, values in per_model.items():
        if values:
            n = len(values)
            result[model] = {
                "min": min(values),
                "max": max(values),
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "std": statistics.stdev(values) if n >= 2 else 0.0,
                "count": n,
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
    Returns {mean, median, std, count, positive, negative, tie} or None when
    fewer than two tasks have paired data.

    The three count buckets partition the paired tasks exactly:
      - ``positive``: model_b strictly better (diff > 0)
      - ``negative``: model_b strictly worse  (diff < 0)
      - ``tie``:      equal passrate          (diff == 0)
    so ``positive + negative + tie == count`` always holds. Ties get their
    own bucket rather than folding into ``negative``; the delivery red-line
    ``claude_not_better`` corresponds to ``negative + tie``.
    """
    diffs: list[float] = []
    for tid in task_ids:
        info = state.get(tid, "finalize")
        a_val = info.get(f"{model_a}_passrate")
        b_val = info.get(f"{model_b}_passrate")
        if (
            isinstance(a_val, (int, float))
            and isinstance(b_val, (int, float))
        ):
            # Round to 4 decimals (the storage precision) so that
            # passrates equal at the displayed level produce diff == 0
            # instead of a floating-point artefact like 5.55e-17.
            diffs.append(round(float(b_val), 4) - round(float(a_val), 4))

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
                "tie": int(d == 0),
            }
        return None

    positive = sum(1 for d in diffs if d > 0)
    negative = sum(1 for d in diffs if d < 0)
    tie = sum(1 for d in diffs if d == 0)
    return {
        "mean": statistics.mean(diffs),
        "median": statistics.median(diffs),
        "std": statistics.stdev(diffs),
        "count": len(diffs),
        "positive": positive,
        "negative": negative,
        "tie": tie,
    }


def _collect_timing_stats(
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> dict[str, dict[str, float]]:
    """Gather duration_s from run and score stages per model.

    Returns {"run/model": {min, max, mean, total, count}, "score/model": ...}
    for stage/model combinations that have data.
    """
    result: dict[str, dict[str, float]] = {}
    for stage in ("run", "score"):
        for model in models:
            values: list[float] = []
            for tid in task_ids:
                info = state.get(tid, stage, model)
                dur = info.get("duration_s")
                if isinstance(dur, (int, float)) and dur > 0:
                    values.append(float(dur))
            if values:
                result[f"{stage}/{model}"] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": statistics.mean(values),
                    "total": sum(values),
                    "count": len(values),
                }
    return result


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string (e.g. '2m30s', '45s')."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = int(seconds) // 60
    secs = int(seconds) % 60
    if minutes < 60:
        return f"{minutes}m{secs:02d}s"
    hours = minutes // 60
    mins = minutes % 60
    return f"{hours}h{mins:02d}m"


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


def _find_slowest(
    state: PipelineState,
    task_ids: list[str],
    models: list[str],
) -> tuple[str, str, float] | None:
    """Return (task_id, model, duration_s) for the slowest run, or None."""
    best_tid = ""
    best_model = ""
    best_dur = 0.0
    for tid in task_ids:
        for model in models:
            info = state.get(tid, "run", model)
            dur = info.get("duration_s")
            if isinstance(dur, (int, float)) and dur > best_dur:
                best_tid, best_model, best_dur = tid, model, float(dur)
    if best_dur > 0:
        return best_tid, best_model, best_dur
    return None


def _print_table(
    stage_rows: list[dict[str, object]],
    passrate_stats: dict[str, dict[str, float]],
    passrate_diff: dict[str, float] | None,
    bottleneck: tuple[str, int],
    model_a: str,
    model_b: str,
    timing_stats: dict[str, dict[str, float]] | None = None,
    slowest: tuple[str, str, float] | None = None,
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
        print(f"\n{'Model':<10} {'Min':>8} {'Max':>8} {'Mean':>8} {'Med':>8} {'Std':>8} {'Count':>6}")
        print("-" * 62)
        for model, s in passrate_stats.items():
            print(
                f"{model:<10} "
                f"{s['min']:>8.4f} "
                f"{s['max']:>8.4f} "
                f"{s['mean']:>8.4f} "
                f"{s['median']:>8.4f} "
                f"{s['std']:>8.4f} "
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
              f"{model_b}={model_a}: {int(passrate_diff['tie'])}   "
              f"{model_b}<{model_a}: {int(passrate_diff['negative'])}")

    # Duration summary per stage/model
    if timing_stats is not None:
        hdr = f"{'Stage':<18} {'Min':>8} {'Max':>8} {'Mean':>8} {'Total':>9} {'Count':>6}"
        print(f"\nRun duration\n{hdr}")
        print("-" * len(hdr))
        if timing_stats:
            for key, d in timing_stats.items():
                print(
                    f"{key:<18} "
                    f"{_fmt_duration(d['min']):>8} "
                    f"{_fmt_duration(d['max']):>8} "
                    f"{_fmt_duration(d['mean']):>8} "
                    f"{_fmt_duration(d['total']):>9} "
                    f"{int(d['count']):>6}"
                )
        else:
            print("  N/A")

    # Bottleneck (stage with most failures)
    stage_name, failed_count = bottleneck
    if failed_count > 0:
        print(f"\nBottleneck: {stage_name} ({failed_count} task(s) failed)")
    else:
        print("\nNo failures. All stages complete or pending.")

    if timing_stats is not None:
        if slowest:
            tid, model, dur = slowest
            print(f"Slowest: {tid}/{model} ({_fmt_duration(dur)})")
        else:
            print("Slowest: N/A")

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
            if isinstance(pr, (int, float)):
                task_detail[f"{model}_passrate"] = round(float(pr), 4)
        # Attach duration_s from run and score stages when available.
        for stage in ("run", "score"):
            for model in models:
                info = state.get(tid, stage, model)
                dur = info.get("duration_s")
                if isinstance(dur, (int, float)) and dur > 0:
                    task_detail[f"{stage}/{model}_duration_s"] = round(float(dur), 1)
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
    timing_stats: dict[str, dict[str, float]] | None = None,
    slowest: tuple[str, str, float] | None = None,
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
                "median": round(s["median"], 4),
                "std": round(s["std"], 4),
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
            "tie": int(passrate_diff["tie"]),
        }

    payload: dict[str, object] = {"summary": summary, "per_task": per_task}
    if timing_stats is not None:
        slowest_task = None
        if slowest:
            slowest_task = {"task": slowest[0], "model": slowest[1], "duration_s": round(slowest[2], 1)}
        payload["timing"] = {
            "per_stage": {
                key: {
                    "min": round(d["min"], 1),
                    "max": round(d["max"], 1),
                    "mean": round(d["mean"], 1),
                    "total": round(d["total"], 1),
                    "count": int(d["count"]),
                }
                for key, d in timing_stats.items()
            },
            "slowest_task": slowest_task,
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def _tokenize_filter(expr: str) -> list[tuple[str, Any]]:
    """Tokenize a filter expression into (type, value) pairs.

    Supported syntax:
      - Comparisons: ``field = 'value'``, ``field < 0.5``, etc.
      - Operators: ``=``, ``!=``, ``<``, ``>``, ``<=``, ``>=``
      - Logic: ``AND``, ``OR`` (case-insensitive)
      - Grouping: ``(`` / ``)``
      - Values: single- or double-quoted strings, integers, floats
      - Fields: ``[a-zA-Z_][a-zA-Z0-9_]*``
    """
    tokens: list[tuple[str, Any]] = []
    i = 0
    while i < len(expr):
        ch = expr[i]

        if ch in (' ', '\t', '\n'):
            i += 1
            continue

        if ch == '(':
            tokens.append(('LPAREN', '('))
            i += 1
            continue
        if ch == ')':
            tokens.append(('RPAREN', ')'))
            i += 1
            continue

        if ch in ('"', "'"):
            quote = ch
            j = i + 1
            while j < len(expr) and expr[j] != quote:
                if expr[j] == '\\':
                    j += 1
                j += 1
            if j >= len(expr):
                raise ValueError(f"Unterminated string literal at position {i}")
            tokens.append(('STRING', expr[i + 1:j]))
            i = j + 1
            continue

        if ch in ('!', '<', '>', '='):
            if i + 1 < len(expr) and expr[i + 1] == '=':
                tokens.append(('OP', expr[i:i + 2]))
                i += 2
            else:
                if ch == '=':
                    tokens.append(('OP', '='))
                else:
                    tokens.append(('OP', ch))
                i += 1
            continue

        if ch.isdigit() or (ch == '-' and i + 1 < len(expr) and expr[i + 1].isdigit()):
            j = i
            if ch == '-':
                j += 1
            while j < len(expr) and expr[j].isdigit():
                j += 1
            if j < len(expr) and expr[j] == '.':
                j += 1
                while j < len(expr) and expr[j].isdigit():
                    j += 1
                tokens.append(('NUMBER', float(expr[i:j])))
            else:
                tokens.append(('NUMBER', int(expr[i:j])))
            i = j
            continue

        if ch.isalpha() or ch == '_':
            j = i
            while j < len(expr) and (expr[j].isalnum() or expr[j] == '_'):
                j += 1
            word = expr[i:j]
            upper = word.upper()
            if upper == 'AND':
                tokens.append(('AND', 'AND'))
            elif upper == 'OR':
                tokens.append(('OR', 'OR'))
            else:
                tokens.append(('FIELD', word))
            i = j
            continue

        raise ValueError(f"Unexpected character {ch!r} at position {i}")

    return tokens


def _parse_filter_expr(expr: str) -> Callable[[dict[str, Any]], bool]:
    """Parse a filter expression into a callable predicate.

    Example::

        fn = _parse_filter_expr("task_type = 'bug-fix' AND qwen_passrate < 0.5")
        fn({"task_type": "bug-fix", "qwen_passrate": 0.3})  # True

    Grammar::

        expr       → or_expr
        or_expr    → and_expr ( 'OR' and_expr )*
        and_expr   → atom ( 'AND' atom )*
        atom       → '(' or_expr ')' | comparison
        comparison → FIELD OP VALUE
    """
    tokens = _tokenize_filter(expr)
    if not tokens:
        raise ValueError("Empty filter expression")

    fn, pos = _parse_or_expr(tokens, 0)
    if pos < len(tokens):
        raise ValueError(f"Unexpected token at position {pos}: {tokens[pos][1]!r}")
    return fn


def _parse_or_expr(tokens: list[tuple[str, Any]], pos: int) -> tuple[Callable, int]:
    fn, pos = _parse_and_expr(tokens, pos)
    fns = [fn]
    while pos < len(tokens) and tokens[pos][0] == 'OR':
        pos += 1
        fn, pos = _parse_and_expr(tokens, pos)
        fns.append(fn)
    if len(fns) == 1:
        return fns[0], pos
    captured = list(fns)
    return lambda record, _fns=captured: any(f(record) for f in _fns), pos


def _parse_and_expr(tokens: list[tuple[str, Any]], pos: int) -> tuple[Callable, int]:
    fn, pos = _parse_atom(tokens, pos)
    fns = [fn]
    while pos < len(tokens) and tokens[pos][0] == 'AND':
        pos += 1
        fn, pos = _parse_atom(tokens, pos)
        fns.append(fn)
    if len(fns) == 1:
        return fns[0], pos
    captured = list(fns)
    return lambda record, _fns=captured: all(f(record) for f in _fns), pos


def _parse_atom(tokens: list[tuple[str, Any]], pos: int) -> tuple[Callable, int]:
    if pos >= len(tokens):
        raise ValueError("Unexpected end of expression")

    if tokens[pos][0] == 'LPAREN':
        pos += 1
        fn, pos = _parse_or_expr(tokens, pos)
        if pos >= len(tokens) or tokens[pos][0] != 'RPAREN':
            raise ValueError("Missing closing ')'")
        return fn, pos + 1

    if pos + 2 >= len(tokens):
        raise ValueError("Incomplete comparison at end of expression")
    if tokens[pos][0] != 'FIELD':
        raise ValueError(f"Expected field name, got {tokens[pos][1]!r}")
    if tokens[pos + 1][0] != 'OP':
        raise ValueError(f"Expected operator, got {tokens[pos + 1][1]!r}")

    field = tokens[pos][1]
    if field not in _FILTER_FIELD_WHITELIST:
        raise ValueError(
            f"Invalid filter field {field!r}. "
            f"Allowed: {', '.join(sorted(_FILTER_FIELD_WHITELIST))}"
        )
    op = tokens[pos + 1][1]
    value = tokens[pos + 2][1]
    pos += 3
    return _make_comparison(field, op, value), pos


def _make_comparison(field: str, op: str, value: Any) -> Callable[[dict[str, Any]], bool]:
    """Build a single-field comparison predicate."""
    def predicate(record: dict[str, Any]) -> bool:
        record_value = record.get(field)
        # None / empty-string → "no data": never match any comparison.
        # Covers relative_gain=None (qwen=0 / missing passrate),
        # finalize_status="" (task not yet finalized), etc.
        if record_value is None or record_value == "":
            return False

        compare_value = value

        # Bool coercion: allow ``threshold_ok = True`` where the value
        # was parsed as the FIELD token "True".
        if isinstance(record_value, bool) and isinstance(value, str):
            lower = value.lower()
            if lower == 'true':
                compare_value = True
            elif lower == 'false':
                compare_value = False

        # Numeric coercion when the literal is a number.
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                record_value = float(record_value)
                compare_value = float(value)
            except (ValueError, TypeError):
                pass

        match op:
            case '=':
                return record_value == compare_value
            case '!=':
                return record_value != compare_value
            case '<':
                return record_value < compare_value
            case '>':
                return record_value > compare_value
            case '<=':
                return record_value <= compare_value
            case '>=':
                return record_value >= compare_value
            case _:
                raise ValueError(f"Unknown operator: {op!r}")

    return predicate


def _build_task_records(
    config: BatchConfig,
    tasks: list,
    state: PipelineState,
    models: list[str],
) -> list[dict[str, Any]]:
    """Build flat per-task records merging TaskConfig fields with finalize state."""
    records: list[dict[str, Any]] = []
    for task in tasks:
        fin = state.get(task.id, "finalize")
        record: dict[str, Any] = {
            "id": task.id,
            "task_type": task.task_type,
            "domain": task.domain,
            "language": task.language,
            "bad_pattern": task.bad_pattern,
            "task_title": task.task_title,
        }
        for model in models:
            pr = fin.get(f"{model}_passrate")
            if isinstance(pr, (int, float)):
                record[f"{model}_passrate"] = round(float(pr), 4)
            else:
                record[f"{model}_passrate"] = None
        record["finalize_status"] = fin.get("status", "")
        record["threshold_ok"] = bool(fin.get("threshold_ok", False))
        # Derived metric: relative_gain = (claude - qwen) / qwen
        # Mirrors the logic in config.check_passrate_thresholds:
        #   - both passrates present and qwen > 0  → compute
        #   - qwen == 0                           → None (data incomplete)
        #   - either passrate missing             → None
        qwen_pr = record.get("qwen_passrate")
        claude_pr = record.get("claude_passrate")
        if (
            isinstance(qwen_pr, (int, float))
            and isinstance(claude_pr, (int, float))
        ):
            if qwen_pr > 0:
                record["relative_gain"] = round(
                    (float(claude_pr) - float(qwen_pr)) / float(qwen_pr), 4,
                )
            else:
                # qwen == 0: consistent with red-line validation — leave empty.
                record["relative_gain"] = None
        else:
            record["relative_gain"] = None

        # 交付红线指标 — 逐条对应 config.check_passrate_thresholds 的 4 项检查
        has_qwen = isinstance(qwen_pr, (int, float))
        has_claude = isinstance(claude_pr, (int, float))
        record["qwen_over_threshold"] = (
            has_qwen and float(qwen_pr) >= THRESHOLD_QWEN_MAX
        )
        record["claude_under_threshold"] = (
            has_claude and float(claude_pr) <= THRESHOLD_CLAUDE_MIN
        )
        record["claude_not_better"] = (
            has_qwen and has_claude and float(claude_pr) <= float(qwen_pr)
        )
        # gain_below_threshold: 当 qwen > 0 时检查 relative_gain；
        # 当 qwen == 0 时检查 claude_passrate 是否低于增益阈值
        # （与 check_passrate_thresholds 的 qwen==0 分支保持一致）
        if has_qwen and has_claude:
            if float(qwen_pr) > 0:
                rg = record.get("relative_gain")
                record["gain_below_threshold"] = (
                    isinstance(rg, (int, float))
                    and float(rg) <= THRESHOLD_RELATIVE_GAIN_MIN
                )
            else:
                record["gain_below_threshold"] = (
                    float(claude_pr) < THRESHOLD_RELATIVE_GAIN_MIN
                )
        else:
            record["gain_below_threshold"] = False

        records.append(record)
    return records


def _apply_filter(
    records: list[dict[str, Any]],
    filter_fn: Callable[[dict[str, Any]], bool],
) -> list[dict[str, Any]]:
    """Return only records that satisfy the filter predicate."""
    return [r for r in records if filter_fn(r)]


def _group_by_fields(
    records: list[dict[str, Any]],
    group_fields: list[str],
    models: list[str],
) -> list[dict[str, Any]]:
    """Group records independently per field and compute passrate stats.

    Returns a list of grouping dicts, one per field, each containing::

        {
            "field": "task_type",
            "count": 59,
            "groups": [
                {
                    "group_value": "bug-fix",
                    "count": 12,
                    "passrate_stats": {
                        "qwen":  {"min": ..., "max": ..., "mean": ..., "count": ...},
                        "claude": {...},
                    },
                },
                ...
            ],
        }
    """
    groupings: list[dict[str, Any]] = []
    for field in group_fields:
        buckets: dict[Any, list[dict[str, Any]]] = {}
        for record in records:
            value = record.get(field, "")
            buckets.setdefault(value, []).append(record)

        groups: list[dict[str, Any]] = []
        for value, bucket_records in buckets.items():
            entry: dict[str, Any] = {
                "group_value": value,
                "count": len(bucket_records),
                "passrate_stats": {},
            }
            for model in models:
                values = [
                    r[f"{model}_passrate"]
                    for r in bucket_records
                    if r.get(f"{model}_passrate") is not None
                ]
                if values:
                    n = len(values)
                    entry["passrate_stats"][model] = {
                        "min": min(values),
                        "max": max(values),
                        "mean": statistics.mean(values),
                        "median": statistics.median(values),
                        "std": statistics.stdev(values) if n >= 2 else 0.0,
                        "count": n,
                    }
            groups.append(entry)
        groupings.append({
            "field": field,
            "count": len(records),
            "groups": groups,
        })
    return groupings


def _format_pr(stats: dict[str, Any] | None) -> str:
    """Format a passrate stats dict as a one-line summary."""
    if not stats:
        return "N/A"
    return (f"min={stats['min']:.4f}  max={stats['max']:.4f}  "
            f"mean={stats['mean']:.4f}  med={stats['median']:.4f}  "
            f"std={stats['std']:.4f}  n={stats['count']}")


def show_stats(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
    fmt: str = "table",
    timing: bool = False,
    filter_expr: str | None = None,
    group_by: str | None = None,
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
    state = PipelineState(config.state_path)

    if not tasks:
        print("No tasks found in delivery manifest or tasks.toml.")
        return False

    tids = [t.id for t in tasks]

    # --- Filter & Group-by ------------------------------------------------
    task_records: list[dict[str, Any]] | None = None
    if filter_expr or group_by:
        task_records = _build_task_records(config, tasks, state, models)

    if filter_expr:
        filter_fn = _parse_filter_expr(filter_expr)
        total_before = len(task_records)  # type: ignore[arg-type]
        task_records = _apply_filter(task_records, filter_fn)  # type: ignore[arg-type]
        tids = [r["id"] for r in task_records]
        if not tids:
            if fmt == "json":
                print(json.dumps({
                    "filter": filter_expr,
                    "total_before_filter": total_before,
                    "total_after_filter": 0,
                    "message": "No tasks match the filter",
                }, indent=2, ensure_ascii=False))
            else:
                print(f"\nNo tasks match filter: {filter_expr}")
                print(f"({total_before} task(s) before filter, 0 after)")
            return True

    if group_by:
        group_fields = [f.strip() for f in group_by.split(",") if f.strip()]
        if task_records and group_fields:
            valid_fields = set(task_records[0].keys())
            for gf in group_fields:
                if gf not in valid_fields:
                    print(f"Invalid group-by field: {gf!r}. Valid: {sorted(valid_fields)}")
                    return False
        groupings = _group_by_fields(task_records, group_fields, models)  # type: ignore[arg-type]

        stage_rows = _collect_stage_counts(state, tids, models)
        all_ok = all(
            int(row["failed"]) == 0 and int(row["pending"]) == 0  # type: ignore[arg-type]
            for row in stage_rows
        )

        if fmt == "json":
            json_groupings: list[dict[str, Any]] = []
            for g in groupings:
                json_groupings.append({
                    "group_key": g["field"],
                    "count": g["count"],
                    "groups": [
                        {
                            "group_value": grp["group_value"],
                            "count": grp["count"],
                            "passrate_stats": {
                                m: {
                                    "min": round(s["min"], 4),
                                    "max": round(s["max"], 4),
                                    "mean": round(s["mean"], 4),
                                    "median": round(s["median"], 4),
                                    "std": round(s["std"], 4),
                                    "count": s["count"],
                                }
                                for m, s in grp["passrate_stats"].items()
                            },
                        }
                        for grp in g["groups"]
                    ],
                })
            payload: dict[str, Any] = {"groupings": json_groupings}
            if filter_expr:
                payload["filter"] = filter_expr
                payload["total_before_filter"] = total_before
                payload["total_after_filter"] = len(task_records)
            payload["summary"] = {
                "total": len(task_records),
                "all_ok": all_ok,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            total_tasks = len(task_records)  # type: ignore[arg-type]
            for gi, g in enumerate(groupings):
                if gi > 0:
                    print()
                hdr = f"\n{'=' * 70}"
                print(hdr)
                if filter_expr:
                    print(f"  PASSRATE BY {g['field'].upper()}"
                          f"  (filtered {total_tasks}/{total_before})")
                else:
                    print(f"  PASSRATE BY {g['field'].upper()}")
                print(f"{'=' * 70}")
                col_hdr = f"  {'Group':<20} {'Model':<10} {'Min':>8} {'Max':>8} {'Mean':>8} {'Med':>8} {'Std':>8} {'Count':>6}"
                print(col_hdr)
                print(f"  {'-' * 82}")
                for grp in g["groups"]:
                    value_str = str(grp["group_value"]) if grp["group_value"] != "" else "(empty)"
                    label = f"{value_str} ({grp['count']})"
                    first = True
                    for model in models:
                        pr = grp["passrate_stats"].get(model)
                        if first:
                            print(f"  {label:<20} {model:<10} {_format_pr(pr)}")
                            first = False
                        else:
                            print(f"  {'':<20} {model:<10} {_format_pr(pr)}")
            print(f"\n{'=' * 70}")
            if filter_expr:
                print(f"  Total: {total_tasks} task(s) after filter"
                      f" (from {total_before})")
            else:
                print(f"  Total: {total_tasks} task(s)")
            if all_ok:
                print("  All stages OK")
            else:
                print("  WARNING: some stages have failures or pending work")
            print(f"{'=' * 70}")
        return all_ok

    stage_rows = _collect_stage_counts(state, tids, models)
    passrate_stats = _collect_passrate_stats(state, tids, models)
    bottleneck = _find_bottleneck(stage_rows)

    timing_stats: dict[str, dict[str, float]] | None = None
    slowest: tuple[str, str, float] | None = None
    if timing:
        timing_stats = _collect_timing_stats(state, tids, models)
        slowest = _find_slowest(state, tids, models)

    # Passrate diff is meaningful when at least two models are compared.
    model_a = models[0] if models else "qwen"
    model_b = models[1] if len(models) >= 2 else "claude"
    passrate_diff = _collect_passrate_diff(state, tids, model_a, model_b) if model_a != model_b else None

    if fmt == "json":
        per_task = _collect_per_task(state, tids, models)
        _print_json(stage_rows, passrate_stats, passrate_diff, bottleneck, per_task, model_a, model_b, timing_stats, slowest)
    else:
        _print_table(stage_rows, passrate_stats, passrate_diff, bottleneck, model_a, model_b, timing_stats, slowest)

    # Return True only when no failures and nothing pending.
    all_ok = all(
        int(row["failed"]) == 0 and int(row["pending"]) == 0  # type: ignore[arg-type]
        for row in stage_rows
    )
    return all_ok
