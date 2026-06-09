"""collect subcommand: find trajectory JSONL files, copy, and verify."""

from __future__ import annotations

import shutil
from pathlib import Path

from ctpipe.config import BatchConfig, MIN_SALVAGE_LINES, MIN_TRAJECTORY_LINES, TaskConfig, select_delivery_tasks, validate_session_id
from ctpipe.state import PipelineState
from ctpipe.project_hash import project_hash_dir
from ctpipe.trajectory import find_trajectory_for_run, parse_trajectory, trajectory_filename


_SALVAGEABLE_STATUSES = ("running", "error", "failed", "timeout")


def _infer_start_time(run_dir: Path, task_id: str, model_name: str) -> float:
    """Infer approximate start_time from .claude/ file mtimes when state is missing it."""
    claude_dir = run_dir / ".claude"
    if claude_dir.is_dir():
        mtimes = [f.stat().st_mtime for f in claude_dir.rglob("*") if f.is_file()]
        if mtimes:
            approx = min(mtimes)
            print(
                f"  [{task_id}/{model_name}] WARN: missing start_time, "
                f"inferred from .claude/ earliest mtime ({approx:.0f})"
            )
            return approx
    print(f"  [{task_id}/{model_name}] WARN: missing start_time, using epoch as fallback")
    return 0.0


def _infer_session_id(run_dir: Path, start_time: float, task_id: str, model_name: str) -> str:
    """Scan the project hash dir for the newest JSONL and extract its session_id."""
    import json as _json

    proj_dir = project_hash_dir(run_dir)
    if not proj_dir.is_dir():
        return ""

    candidates = [
        (f, f.stat().st_mtime)
        for f in proj_dir.iterdir()
        if f.suffix == ".jsonl" and f.stat().st_mtime > start_time
    ]
    if not candidates:
        return ""

    candidates.sort(key=lambda item: item[1], reverse=True)
    newest = candidates[0][0]

    with newest.open("r", encoding="utf-8", errors="replace") as fh:
        for i, line in enumerate(fh):
            if i >= 30:
                break
            line = line.strip()
            if not line:
                continue
            try:
                obj = _json.loads(line)
            except _json.JSONDecodeError:
                continue
            if isinstance(obj, dict) and obj.get("sessionId"):
                sid = obj["sessionId"]
                print(
                    f"  [{task_id}/{model_name}] WARN: missing session_id, "
                    f"inferred {sid!r} from {newest.name}"
                )
                return sid
    return ""


def collect_single(
    task: TaskConfig,
    model_name: str,
    config: BatchConfig,
    state: PipelineState,
    *,
    salvage: bool = True,
    force: bool = False,
) -> bool:
    run_info = state.get(task.id, "run", model_name)
    run_status = run_info.get("status", "")

    is_salvage = False
    if force:
        if run_status not in ("done", "partial"):
            is_salvage = True
        print(f"  [{task.id}/{model_name}] --force: bypassing start_time/session_id validation")
    elif run_status in ("done", "partial"):
        pass
    elif salvage and run_status in _SALVAGEABLE_STATUSES:
        is_salvage = True
        print(f"  [{task.id}/{model_name}] run interrupted (status={run_status!r}), attempting salvage")
    else:
        print(f"  [{task.id}/{model_name}] run not done (status={run_status!r}), skipping collect")
        return False

    session_id = run_info.get("session_id", "")
    prepare_info = state.get(task.id, "prepare")
    run_dir = Path(prepare_info.get(f"{model_name}_dir", ""))

    if force:
        start_time = 0.0
        session_id = ""
    else:
        start_time = run_info.get("start_time", None)
        if start_time is None or start_time == 0:
            start_time = _infer_start_time(run_dir, task.id, model_name)

        if not session_id and run_dir.is_dir():
            session_id = _infer_session_id(run_dir, start_time, task.id, model_name)

        # Validate session_id format before using in file paths
        if session_id:
            try:
                validate_session_id(session_id)
            except ValueError:
                print(f"  [{task.id}/{model_name}] ERROR: invalid session_id format: {session_id!r}")
                state.set(task.id, "collect", model=model_name, status="failed", error="invalid session_id format")
                return False

    if not run_dir.is_dir():
        print(f"  [{task.id}/{model_name}] ERROR: run dir not found: {run_dir}")
        state.set(
            task.id, "collect", model=model_name,
            status="failed", error="run dir not found",
            recovery=is_salvage or force,
        )
        return False

    jsonl_path = find_trajectory_for_run(run_dir, start_time, session_id or None)
    if not jsonl_path:
        if is_salvage:
            print(f"  [{task.id}/{model_name}] salvage: no JSONL found, nothing to recover")
            state.set(
                task.id, "collect", model=model_name,
                status="skipped", error="salvage: no JSONL",
                recovery=True,
            )
            return False
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
            recovery=is_salvage or force,
        )
        return False

    min_lines = MIN_SALVAGE_LINES if is_salvage else MIN_TRAJECTORY_LINES

    if info.line_count < min_lines or (not is_salvage and not info.models):
        if is_salvage and info.line_count > 0:
            print(
                f"  [{task.id}/{model_name}] salvage: trajectory too short "
                f"(lines={info.line_count}), collecting as partial"
            )
        else:
            print(
                f"  [{task.id}/{model_name}] ERROR: trajectory structurally incomplete — "
                f"lines={info.line_count}, models={info.models}"
            )
            print(f"    Source: {jsonl_path}")
            state.set(
                task.id, "collect", model=model_name,
                status="failed",
                error=f"trajectory incomplete: lines={info.line_count}, models={len(info.models)}",
                recovery=is_salvage or force,
            )
            return False

    if session_id and info.session_id != session_id:
        actual = info.session_id or "(none)"
        if is_salvage:
            print(
                f"  [{task.id}/{model_name}] WARN: session_id mismatch — "
                f"expected {session_id!r}, got {actual} (continuing salvage)"
            )
        else:
            print(
                f"  [{task.id}/{model_name}] ERROR: session_id mismatch — "
                f"expected {session_id!r}, got {actual}"
            )
            print(f"    Source: {jsonl_path}")
            state.set(
                task.id, "collect", model=model_name,
                status="failed",
                error=f"session_id mismatch: expected {session_id}, got {actual}",
                recovery=is_salvage or force,
            )
            return False

    dest_dir = config.delivery_dir / "trajectories" / model_name
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / trajectory_filename(task.id)
    shutil.copy2(jsonl_path, dest_path)

    is_recovery = is_salvage or force
    final_status = "partial" if is_salvage else "done"
    label = "Salvaged" if is_salvage else ("Forced" if force else "Copied")
    print(f"  [{task.id}/{model_name}] {label} {jsonl_path.name} -> {dest_path.name}")
    print(
        f"    session_id={info.session_id}, provider={info.detected_provider}, "
        f"lines={info.line_count}, models={info.models}"
    )

    state.set(
        task.id,
        "collect",
        model=model_name,
        status=final_status,
        recovery=is_recovery,
        salvaged=is_salvage,
        forced=force,
        run_status_at_collect=run_status,
        jsonl_source=str(jsonl_path),
        jsonl_path=dest_path.relative_to(config.delivery_dir).as_posix(),
        session_id=info.session_id,
        model_detected=info.detected_provider,
        line_count=info.line_count,
    )
    return True


def collect_all(config: BatchConfig, task_ids: list[str] | None = None, models: list[str] | None = None, *, salvage: bool = True, force: bool = False) -> None:
    state = PipelineState(config.state_path)
    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _collect_task_model(task: TaskConfig, model_name: str) -> None:
        if not force and state.is_done(task.id, "collect", model_name):
            print(f"[{task.id}/{model_name}] collect already done, skipping")
            return
        print(f"[{task.id}/{model_name}] Collecting trajectory...")
        collect_single(task, model_name, config, state, salvage=salvage, force=force)

    with state.batch():
        with ThreadPoolExecutor(max_workers=config.max_parallel) as executor:
            futures = []
            for task in tasks:
                for model_name in models:
                    futures.append(executor.submit(_collect_task_model, task, model_name))
            for future in as_completed(futures):
                exc = future.exception()
                if exc:
                    print(f"  ERROR in collect: {exc}")

    print("Collect complete.")
