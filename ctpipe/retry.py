"""retry subcommand: auto-retry failed/partial pipeline tasks.

Scans pipeline_state.json for non-success entries, resets their state,
re-executes the corresponding stages, and marks tasks that exceed
--max-retries as permanently_failed.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from ctpipe.config import BatchConfig, MODEL_SPECIFIC_STAGES, TaskConfig, select_delivery_tasks
from ctpipe.state import PipelineState

# Stages that do NOT have per-model entries.
_MODEL_AGNOSTIC_STAGES = ("prepare", "finalize")
# Ordered pipeline stages (no validate — it's a read-only check).
_STAGE_ORDER = ("prepare", "run", "collect", "score", "finalize")
# Status values that trigger a retry.
_RETRYABLE_STATUSES = {"failed", "partial", "draft"}

# Downstream cascade: when a stage fails, its downstream stages
# must also be re-executed because their inputs may have changed.
_DOWNSTREAM: dict[str, list[str]] = {
    "prepare": ["run", "collect", "score", "finalize"],
    "run":     ["collect", "score", "finalize"],
    "collect": ["score", "finalize"],
    "score":   ["finalize"],
    "finalize": [],
}


@dataclass(frozen=True)
class _Entry:
    """A single retryable unit: (task_id, stage, model_or_None)."""
    task_id: str
    stage: str
    model: str | None = None

    @property
    def label(self) -> str:
        if self.model:
            return f"{self.task_id}/{self.stage}/{self.model}"
        return f"{self.task_id}/{self.stage}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _Entry):
            return NotImplemented
        return (self.task_id, self.stage, self.model) == (other.task_id, other.stage, other.model)

    def __hash__(self) -> int:
        return hash((self.task_id, self.stage, self.model))


def _get_status(state: PipelineState, entry: _Entry) -> str:
    """Read the status string for a given entry."""
    info = state.get(entry.task_id, entry.stage, entry.model)
    return info.get("status", "")


def _get_retry_count(state: PipelineState, entry: _Entry) -> int:
    """Read the retry_count for a given entry."""
    info = state.get(entry.task_id, entry.stage, entry.model)
    return info.get("retry_count", 0)


def _find_failed_entries(
    state: PipelineState,
    tasks: list[TaskConfig],
    stages: list[str] | None,
    models: list[str],
) -> list[_Entry]:
    """Scan state for entries with retryable status.

    Returns a list of _Entry objects, ordered by stage order then task_id.
    """
    target_stages = stages or list(_STAGE_ORDER)
    entries: list[_Entry] = []

    for stage in _STAGE_ORDER:
        if stage not in target_stages:
            continue
        for task in tasks:
            if stage in _MODEL_AGNOSTIC_STAGES:
                status = _get_status(state, _Entry(task.id, stage))
                if status in _RETRYABLE_STATUSES:
                    entries.append(_Entry(task.id, stage))
            elif stage in MODEL_SPECIFIC_STAGES:
                for model in models:
                    status = _get_status(state, _Entry(task.id, stage, model))
                    if status in _RETRYABLE_STATUSES:
                        entries.append(_Entry(task.id, stage, model))

    return entries


def _expand_cascade(entries: list[_Entry], models: list[str]) -> list[_Entry]:
    """Add downstream stage entries for each failed entry.

    For model-specific downstream stages, the same model is used.
    Returns a deduplicated, stage-ordered list.
    """
    expanded: dict[str, _Entry] = {}  # key = label -> entry

    for entry in entries:
        expanded[entry.label] = entry
        for downstream_stage in _DOWNSTREAM.get(entry.stage, []):
            if downstream_stage in _MODEL_AGNOSTIC_STAGES:
                ds_entry = _Entry(entry.task_id, downstream_stage)
                expanded[ds_entry.label] = ds_entry
            elif downstream_stage in MODEL_SPECIFIC_STAGES:
                # Carry the model forward from the failed entry.
                if entry.model:
                    ds_entry = _Entry(entry.task_id, downstream_stage, entry.model)
                else:
                    # Model-agnostic upstream (prepare) → all models downstream.
                    for model in models:
                        ds_entry = _Entry(entry.task_id, downstream_stage, model)
                        expanded[ds_entry.label] = ds_entry
                    continue
                expanded[ds_entry.label] = ds_entry

    # Sort by stage order, then task_id, then model.
    stage_rank = {s: i for i, s in enumerate(_STAGE_ORDER)}
    result = sorted(
        expanded.values(),
        key=lambda e: (stage_rank.get(e.stage, 99), e.task_id, e.model or ""),
    )
    return result


def _reset_entries(state: PipelineState, entries: list[_Entry]) -> int:
    """Reset state for all given entries. Returns count of resets."""
    count = 0
    with state.batch():
        for entry in entries:
            if state.reset(entry.task_id, entry.stage, entry.model):
                count += 1
    return count


def _group_by_stage(entries: list[_Entry]) -> dict[str, list[_Entry]]:
    """Group entries by stage, preserving stage order."""
    groups: dict[str, list[_Entry]] = {}
    for stage in _STAGE_ORDER:
        groups[stage] = []
    for entry in entries:
        groups.setdefault(entry.stage, []).append(entry)
    return {k: v for k, v in groups.items() if v}


def _task_ids_from_entries(entries: list[_Entry]) -> list[str]:
    """Extract unique task IDs from entries, preserving order."""
    seen: set[str] = set()
    result: list[str] = []
    for e in entries:
        if e.task_id not in seen:
            seen.add(e.task_id)
            result.append(e.task_id)
    return result


def _models_from_entries(entries: list[_Entry], all_models: list[str]) -> list[str]:
    """Extract unique model names from entries."""
    seen: set[str] = set()
    for e in entries:
        if e.model:
            seen.add(e.model)
    return [m for m in all_models if m in seen] or list(all_models)


async def _execute_stage(
    stage: str,
    config: BatchConfig,
    task_ids: list[str],
    models: list[str],
    turn_timeout: int,
    total_timeout: int,
    state: PipelineState,
) -> None:
    """Execute a single pipeline stage for the given task IDs and models."""
    if stage == "prepare":
        from ctpipe.prepare import prepare
        prepare(config, task_ids)

    elif stage == "run":
        from ctpipe.run import run_all
        await run_all(config, task_ids, models, turn_timeout, total_timeout)

    elif stage == "collect":
        from ctpipe.collect import collect_all
        collect_all(config, task_ids, models)

    elif stage == "score":
        from ctpipe.score import score_all
        await score_all(config, task_ids, models)

    elif stage == "finalize":
        from ctpipe.finalize import finalize
        finalize(config, task_ids, models)

    state.reload()


def _mark_permanently_failed(
    state: PipelineState,
    entries: list[_Entry],
) -> list[_Entry]:
    """Mark entries that exceeded max retries as permanently_failed.

    Returns the list of entries that were marked.
    """
    marked: list[_Entry] = []
    for entry in entries:
        status = _get_status(state, entry)
        if status not in _RETRYABLE_STATUSES:
            continue
        info = state.get(entry.task_id, entry.stage, entry.model)
        retry_count = info.get("retry_count", 0)
        last_error = info.get("error", "")
        state.set(
            entry.task_id, entry.stage, entry.model,
            status="permanently_failed",
            retry_count=retry_count,
            error=last_error or f"exceeded max retries",
        )
        marked.append(entry)
    return marked


def _increment_retry_count(
    state: PipelineState,
    entries: list[_Entry],
    saved_counts: dict[_Entry, int] | None = None,
) -> None:
    """Increment retry_count for all entries before re-execution.

    Args:
        saved_counts: Pre-read retry counts (read before reset).
            If None, reads current state (may be 0 after reset).
    """
    for entry in entries:
        if saved_counts is not None:
            retry_count = saved_counts.get(entry, 0)
        else:
            info = state.get(entry.task_id, entry.stage, entry.model)
            retry_count = info.get("retry_count", 0)
        state.set(entry.task_id, entry.stage, entry.model,
                  status="pending", retry_count=retry_count + 1)


def _print_round_summary(
    round_num: int,
    max_retries: int,
    entries: list[_Entry],
    stage_groups: dict[str, list[_Entry]],
) -> None:
    """Print what will be retried in this round."""
    print(f"\n{'=' * 60}")
    print(f"RETRY ROUND {round_num}/{max_retries}")
    print(f"{'=' * 60}")
    print(f"  {len(entries)} entries to retry across {len(stage_groups)} stage(s):")
    for stage, group in stage_groups.items():
        labels = ", ".join(e.label for e in group[:5])
        if len(group) > 5:
            labels += f", ... (+{len(group) - 5} more)"
        print(f"    {stage}: {labels}")
    print()


def _print_results(
    state: PipelineState,
    entries: list[_Entry],
    perm_failed: list[_Entry],
    errors: list[tuple[_Entry, str]],
) -> None:
    """Print a summary of retry results."""
    succeeded = 0
    still_failed = 0
    for entry in entries:
        status = _get_status(state, entry)
        if status == "done":
            succeeded += 1
        else:
            still_failed += 1

    print(f"\n{'=' * 60}")
    print("RETRY RESULTS")
    print(f"{'=' * 60}")
    print(f"  Succeeded : {succeeded}/{len(entries)}")
    if still_failed:
        print(f"  Still failed: {still_failed}")
    if perm_failed:
        print(f"\n  PERMANENTLY FAILED ({len(perm_failed)} entries, exceeded max retries):")
        for entry in perm_failed:
            info = state.get(entry.task_id, entry.stage, entry.model)
            err = info.get("error", "unknown")
            print(f"    {entry.label} — {err}")
    if errors:
        print(f"\n  EXCEPTIONS during retry ({len(errors)}):")
        for entry, err in errors:
            print(f"    {entry.label} — {err}")
    print(f"{'=' * 60}")


async def retry(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    stages: list[str] | None = None,
    models: list[str] | None = None,
    max_retries: int = 2,
    turn_timeout: int = 900,
    total_timeout: int = 3600,
    dry_run: bool = False,
    cascade: bool = True,
) -> bool:
    """Auto-retry failed/partial/draft pipeline tasks.

    Returns True if all tasks are in 'done' state after retry, False otherwise.
    """
    models = models or ["qwen", "claude"]

    delivery_dir = config.delivery_dir
    if not delivery_dir.exists():
        print(f"Delivery directory not found: {delivery_dir}")
        print("Run 'ctpipe prepare' first to initialize the pipeline.")
        return False

    state = PipelineState(config.state_path)
    tasks = select_delivery_tasks(config, task_ids)

    if not tasks:
        print("No tasks found.")
        return False

    for round_num in range(1, max_retries + 1):
        state.reload()
        failed_entries = _find_failed_entries(state, tasks, stages, models)

        if not failed_entries:
            if round_num == 1:
                print("Nothing to retry — all matching tasks are done or pending.")
            else:
                print(f"\nRound {round_num}: No more failed tasks to retry.")
            break

        # Separate entries that exceeded max retries.
        exhausted: list[_Entry] = []
        retryable: list[_Entry] = []
        for entry in failed_entries:
            retry_count = _get_retry_count(state, entry)
            if retry_count >= max_retries:
                exhausted.append(entry)
            else:
                retryable.append(entry)

        # Mark exhausted entries as permanently_failed.
        perm_failed = _mark_permanently_failed(state, exhausted) if exhausted else []
        for entry in perm_failed:
            print(f"  MARKED permanently_failed: {entry.label} "
                  f"(retried {max_retries} times)")

        if not retryable:
            print(f"\nAll {len(exhausted)} remaining failures exceeded max retries "
                  f"({max_retries}). Marked as permanently_failed.")
            break

        # Expand cascade.
        if cascade:
            retryable = _expand_cascade(retryable, models)

        stage_groups = _group_by_stage(retryable)

        if dry_run:
            _print_round_summary(round_num, max_retries, retryable, stage_groups)
            print("  [DRY RUN] No changes made.")
            if round_num == 1:
                break
            continue

        _print_round_summary(round_num, max_retries, retryable, stage_groups)

        # Save retry counts BEFORE reset (reset deletes entries).
        saved_counts = {entry: _get_retry_count(state, entry) for entry in retryable}

        # Reset all entries.
        reset_count = _reset_entries(state, retryable)
        print(f"  Reset {reset_count} state entries.")

        # Set retry_count = old + 1 (using pre-reset counts).
        _increment_retry_count(state, retryable, saved_counts)

        # Execute each stage in order.  Each task within a stage is isolated:
        # one task's exception does NOT block others.
        all_errors: list[tuple[_Entry, str]] = []

        for stage, group in stage_groups.items():
            s_task_ids = _task_ids_from_entries(group)
            s_models = _models_from_entries(group, models)
            stage_label = stage.upper()

            print(f"\n--- Executing {stage_label} ({len(group)} entries) ---")
            start = time.time()

            try:
                await _execute_stage(
                    stage, config, s_task_ids, s_models,
                    turn_timeout, total_timeout, state,
                )
            except Exception as exc:
                # Stage-level crash: mark ALL entries in this stage as failed.
                err_msg = f"stage {stage} crashed: {exc}"
                print(f"  ERROR: {err_msg}")
                for entry in group:
                    all_errors.append((entry, err_msg))
                    state.set(
                        entry.task_id, entry.stage, entry.model,
                        status="failed",
                        error=err_msg,
                        retry_count=_get_retry_count(state, entry),
                    )
                continue

            elapsed = time.time() - start

            # Per-entry result check: individual tasks may have failed.
            for entry in group:
                status = _get_status(state, entry)
                if status in _RETRYABLE_STATUSES:
                    err_info = state.get(entry.task_id, entry.stage, entry.model)
                    err_msg = err_info.get("error", f"status={status}")
                    all_errors.append((entry, err_msg))

            print(f"  {stage_label} completed in {elapsed:.0f}s")

        # Print per-round result.
        _print_results(state, retryable, perm_failed, all_errors)

    # Final state check.
    state.reload()
    remaining = _find_failed_entries(state, tasks, stages, models)
    # Also check for permanently_failed.
    perm_count = 0
    for task in tasks:
        for stage in _STAGE_ORDER:
            if stage in _MODEL_AGNOSTIC_STAGES:
                if _get_status(state, _Entry(task.id, stage)) == "permanently_failed":
                    perm_count += 1
            else:
                for model in models:
                    if _get_status(state, _Entry(task.id, stage, model)) == "permanently_failed":
                        perm_count += 1

    if perm_count:
        print(f"\n{perm_count} entries marked as permanently_failed.")
    if remaining:
        print(f"\n{len(remaining)} entries still need attention.")
        return False
    if perm_count:
        return False
    return True
