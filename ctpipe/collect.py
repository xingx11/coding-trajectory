"""collect subcommand: find trajectory JSONL files, copy, and verify."""

from __future__ import annotations

import shutil
from pathlib import Path

from ctpipe.config import BatchConfig, TaskConfig, select_delivery_tasks
from ctpipe.state import PipelineState
from ctpipe.trajectory import find_trajectory_for_run, parse_trajectory, trajectory_filename


def collect_single(
    task: TaskConfig,
    model_name: str,
    config: BatchConfig,
    state: PipelineState,
) -> bool:
    run_info = state.get(task.id, "run", model_name)
    run_status = run_info.get("status", "")
    if run_status not in ("done", "partial"):
        print(f"  [{task.id}/{model_name}] run not done (status={run_status!r}), skipping collect")
        return False

    session_id = run_info.get("session_id", "")
    start_time = run_info.get("start_time", 0)
    prepare_info = state.get(task.id, "prepare")
    run_dir = Path(prepare_info.get(f"{model_name}_dir", ""))

    if not run_dir.is_dir():
        print(f"  [{task.id}/{model_name}] ERROR: run dir not found: {run_dir}")
        state.set(task.id, "collect", model=model_name, status="failed", error="run dir not found")
        return False

    jsonl_path = find_trajectory_for_run(run_dir, start_time, session_id or None)
    if not jsonl_path:
        print(f"  [{task.id}/{model_name}] ERROR: no JSONL found for run")
        state.set(task.id, "collect", model=model_name, status="failed", error="no JSONL found")
        return False

    info = parse_trajectory(jsonl_path)
    if info.detected_provider not in (model_name, "unknown"):
        print(
            f"  [{task.id}/{model_name}] ERROR: provider mismatch — "
            f"detected {info.detected_provider!r}, expected {model_name!r}"
        )
        print(f"    Models in file: {info.models}")
        print(f"    Source: {jsonl_path}")
        state.set(
            task.id, "collect", model=model_name,
            status="failed",
            error=f"provider mismatch: detected {info.detected_provider}",
        )
        return False

    MIN_TRAJECTORY_LINES = 10
    if info.line_count < MIN_TRAJECTORY_LINES or not info.models:
        print(
            f"  [{task.id}/{model_name}] ERROR: trajectory structurally incomplete — "
            f"lines={info.line_count}, models={info.models}"
        )
        print(f"    Source: {jsonl_path}")
        state.set(
            task.id, "collect", model=model_name,
            status="failed",
            error=f"trajectory incomplete: lines={info.line_count}, models={len(info.models)}",
        )
        return False

    if session_id and info.session_id != session_id:
        actual = info.session_id or "(none)"
        print(
            f"  [{task.id}/{model_name}] ERROR: session_id mismatch — "
            f"expected {session_id!r}, got {actual}"
        )
        print(f"    Source: {jsonl_path}")
        state.set(
            task.id, "collect", model=model_name,
            status="failed",
            error=f"session_id mismatch: expected {session_id}, got {actual}",
        )
        return False

    dest_dir = config.delivery_dir / "trajectories" / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / trajectory_filename(task.id)
    shutil.copy2(jsonl_path, dest_path)

    print(f"  [{task.id}/{model_name}] Copied {jsonl_path.name} -> {dest_path.name}")
    print(
        f"    session_id={info.session_id}, provider={info.detected_provider}, "
        f"lines={info.line_count}, models={info.models}"
    )

    state.set(
        task.id,
        "collect",
        model=model_name,
        status="done",
        jsonl_source=str(jsonl_path),
        jsonl_path=dest_path.relative_to(config.delivery_dir).as_posix(),
        session_id=info.session_id,
        model_detected=info.detected_provider,
        line_count=info.line_count,
    )
    return True


def collect_all(config: BatchConfig, task_ids: list[str] | None = None, models: list[str] | None = None) -> None:
    state = PipelineState(config.delivery_dir / "pipeline_state.json")
    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]

    for task in tasks:
        with state.batch():
            for model_name in models:
                if state.is_done(task.id, "collect", model_name):
                    print(f"[{task.id}/{model_name}] collect already done, skipping")
                    continue
                print(f"[{task.id}/{model_name}] Collecting trajectory...")
                collect_single(task, model_name, config, state)

    print("Collect complete.")
