"""run subcommand: execute claude -p with multi-turn follow-ups via --resume."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

from ctpipe.config import (
    BatchConfig,
    ModelConfig,
    TaskConfig,
    build_validated_env,
    select_delivery_tasks,
)
from ctpipe.state import PipelineState
from ctpipe.trajectory import find_trajectory_for_run, parse_trajectory


def _sanitize_run_dir(run_dir: Path) -> None:
    """Remove untrusted config from cloned repos before execution.

    Cloned repos may contain .claude/ directories or CLAUDE.md files
    that could execute arbitrary commands when --dangerously-skip-permissions
    is used. Remove them, then rebuild the pipeline's own settings.
    """

    # Remove untrusted .claude/ directory
    claude_dir = run_dir / ".claude"
    if claude_dir.is_dir():
        import shutil
        shutil.rmtree(claude_dir, ignore_errors=True)
        print(f"  [security] Removed untrusted .claude/ from {run_dir.name}")

    # Remove CLAUDE.md files that could inject instructions
    for md_name in ("CLAUDE.md", "claude.md", ".claudeignore"):
        md_path = run_dir / md_name
        if md_path.exists() and not md_path.is_symlink():
            md_path.unlink()
            print(f"  [security] Removed untrusted {md_name} from {run_dir.name}")

    # Rebuild pipeline's trusted settings
    claude_dir.mkdir(parents=True, exist_ok=True)
    from ctpipe.prepare import SETTINGS_LOCAL
    (claude_dir / "settings.local.json").write_text(
        json.dumps(SETTINGS_LOCAL, indent=2), encoding="utf-8",
    )


@dataclass
class TurnResult:
    turn: int
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    session_id: str = ""


def _parse_session_id(stdout: str) -> str:
    try:
        data = json.loads(stdout)
        return data.get("session_id", "") or data.get("sessionId", "")
    except (json.JSONDecodeError, TypeError):
        pass

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except (json.JSONDecodeError, TypeError):
            continue
        session_id = data.get("session_id", "") or data.get("sessionId", "")
        if session_id:
            return session_id
    return ""


async def _run_claude_p(
    prompt: str,
    env: dict[str, str],
    cwd: Path,
    model: str | None = None,
    resume_session: str | None = None,
    timeout: int = 900,
) -> TurnResult:
    cmd = [
        "claude",
        "-p",
        "--output-format",
        "json",
        "--dangerously-skip-permissions",
        "--setting-sources", "local",
    ]
    if model:
        cmd += ["--model", model]
    if resume_session:
        cmd += ["--resume", resume_session]

    started_at = time.time()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(cwd),
        env=env,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    async def _heartbeat():
        while True:
            await asyncio.sleep(30)
            elapsed = int(time.time() - started_at)
            print(f"    ... still running ({elapsed}s elapsed)")

    heartbeat_task = asyncio.create_task(_heartbeat())
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(input=prompt.encode("utf-8")), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        heartbeat_task.cancel()
        try:
            await heartbeat_task
        except asyncio.CancelledError:
            pass
        return TurnResult(
            turn=0,
            exit_code=-1,
            stdout="",
            stderr="TIMEOUT",
            duration_s=time.time() - started_at,
        )
    heartbeat_task.cancel()
    try:
        await heartbeat_task
    except asyncio.CancelledError:
        pass

    stdout = stdout_bytes.decode("utf-8", errors="replace")
    stderr = stderr_bytes.decode("utf-8", errors="replace")
    return TurnResult(
        turn=0,
        exit_code=proc.returncode or 0,
        stdout=stdout,
        stderr=stderr,
        duration_s=time.time() - started_at,
        session_id=_parse_session_id(stdout),
    )


async def run_single(
    task: TaskConfig,
    model_name: str,
    model_config: ModelConfig,
    run_dir: Path,
    prompt: str,
    followups: list[str],
    state: PipelineState,
    turn_timeout: int = 900,
    total_timeout: int = 3600,
) -> dict[str, object]:
    env = build_validated_env(model_config)
    start_time = time.time()
    turns_completed = 0
    session_id = ""
    had_errors = False
    error_count = 0
    all_results: list[dict[str, object]] = []

    print(f"  [{task.id}/{model_name}] Turn 1/{1 + len(followups)}: initial prompt...")
    result = await _run_claude_p(prompt, env, run_dir, model=model_config.model, timeout=turn_timeout)
    result.turn = 1
    turns_completed = 1
    session_id = result.session_id
    had_errors = result.exit_code != 0
    if result.exit_code != 0:
        error_count += 1

    # Fallback: if session_id empty (e.g. timeout killed the process),
    # try recovering from the JSONL file Claude Code already wrote to disk.
    if not session_id:
        traj_path = find_trajectory_for_run(run_dir, start_time)
        if traj_path:
            traj_info = parse_trajectory(traj_path)
            if traj_info.session_id:
                session_id = traj_info.session_id
                print(f"  [{task.id}/{model_name}] Recovered session_id from disk: {session_id}")

    all_results.append({
        "turn": 1,
        "exit_code": result.exit_code,
        "duration_s": round(result.duration_s, 1),
        "session_id": session_id,
    })

    if not session_id:
        error = "could not extract session_id from turn 1"
        print(f"  [{task.id}/{model_name}] ERROR: {error}")
        summary = {
            "status": "failed",
            "session_id": "",
            "turns": turns_completed,
            "duration_s": round(time.time() - start_time, 1),
            "start_time": start_time,
            "turns_detail": all_results,
            "error": error,
        }
        state.set(task.id, "run", model=model_name, **summary)
        return summary

    if result.exit_code != 0:
        print(f"  [{task.id}/{model_name}] Turn 1 exited with {result.exit_code}; continuing with captured session")

    for index, followup in enumerate(followups, start=2):
        elapsed = time.time() - start_time
        if elapsed >= total_timeout:
            print(f"  [{task.id}/{model_name}] Total timeout reached after {turns_completed} turns")
            break

        effective_timeout = min(turn_timeout, max(1, int(total_timeout - elapsed)))
        print(f"  [{task.id}/{model_name}] Turn {index}/{1 + len(followups)}: follow-up...")
        result = await _run_claude_p(
            followup,
            env,
            run_dir,
            model=model_config.model,
            resume_session=session_id,
            timeout=effective_timeout,
        )
        result.turn = index
        turns_completed = index
        had_errors = had_errors or result.exit_code != 0
        if result.exit_code != 0:
            error_count += 1

        all_results.append({
            "turn": index,
            "exit_code": result.exit_code,
            "duration_s": round(result.duration_s, 1),
        })

        if result.exit_code != 0:
            print(f"  [{task.id}/{model_name}] Turn {index} failed (exit={result.exit_code}), continuing...")

    total_duration = time.time() - start_time
    all_turns_done = turns_completed == 1 + len(followups)
    all_turns_errored = error_count == turns_completed
    if all_turns_errored:
        status = "failed"
    elif not all_turns_done or had_errors:
        status = "partial"
    else:
        status = "done"
    summary = {
        "status": status,
        "session_id": session_id,
        "turns": turns_completed,
        "expected_turns": 1 + len(followups),
        "duration_s": round(total_duration, 1),
        "start_time": start_time,
        "turns_detail": all_results,
        "had_errors": had_errors,
    }
    state.set(task.id, "run", model=model_name, **summary)
    label = "FAILED (all turns errored)" if all_turns_errored else f"Done: {turns_completed} turns"
    print(f"  [{task.id}/{model_name}] {label} in {total_duration:.0f}s")
    return summary


async def run_task_model(
    task: TaskConfig,
    config: BatchConfig,
    state: PipelineState,
    model_name: str,
    turn_timeout: int = 900,
    total_timeout: int = 3600,
) -> None:
    prepare_info = state.get(task.id, "prepare")
    if state.is_done(task.id, "run", model_name):
        print(f"[{task.id}/{model_name}] run already done, skipping")
        return

    run_dir = Path(prepare_info.get(f"{model_name}_dir", ""))
    if not run_dir.is_dir():
        print(f"[{task.id}/{model_name}] ERROR: run dir not found: {run_dir}")
        state.set(task.id, "run", model=model_name, status="failed", error="run dir not found")
        return

    # Remove untrusted .claude/ config from cloned repos before execution
    _sanitize_run_dir(run_dir)

    model_config = config.qwen if model_name == "qwen" else config.claude
    prompt = task.prompt_qwen if model_name == "qwen" else task.prompt_claude
    followups = task.followups_qwen if model_name == "qwen" else task.followups_claude

    await run_single(
        task,
        model_name,
        model_config,
        run_dir,
        prompt,
        followups,
        state,
        turn_timeout,
        total_timeout,
    )


async def run_all(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
    turn_timeout: int = 900,
    total_timeout: int = 3600,
    *,
    dry_run: bool = False,
    as_json: bool = False,
) -> dict | None:
    models = models or ["qwen", "claude"]

    if dry_run:
        state = PipelineState(config.state_path)
        tasks = select_delivery_tasks(config, task_ids)
        slots_data = []
        will_run = 0
        will_skip = 0

        if not as_json:
            print("=" * 60)
            print("  DRY RUN: run")
            print("=" * 60)

        _REDACT_KEYS = {"ANTHROPIC_AUTH_TOKEN"}
        for task in tasks:
            for model_name in models:
                already_done = state.is_done(task.id, "run", model_name)
                if already_done:
                    will_skip += 1
                    slots_data.append({
                        "task_id": task.id, "model": model_name, "skip": True,
                    })
                    if not as_json:
                        print(f"\n[{task.id}/{model_name}]  SKIP (already done)")
                    continue

                will_run += 1
                model_config = config.qwen if model_name == "qwen" else config.claude
                prompt = task.prompt_qwen if model_name == "qwen" else task.prompt_claude
                followups = task.followups_qwen if model_name == "qwen" else task.followups_claude
                run_dir = config.runs_root / f"{task.id}-{model_name}" / task.project_subdir

                cmd = [
                    "claude", "-p",
                    "--output-format", "json",
                    "--dangerously-skip-permissions",
                    "--setting-sources", "local",
                ]
                if model_config.model:
                    cmd += ["--model", model_config.model]

                env = build_validated_env(model_config)
                env_masked = {}
                for k in sorted(env):
                    env_masked[k] = "***REDACTED***" if k in _REDACT_KEYS else env[k]

                slot = {
                    "task_id": task.id, "model": model_name, "skip": False,
                    "cmd": cmd, "cwd": str(run_dir),
                    "model_id": model_config.model or None,
                    "prompt_preview": prompt[:100], "prompt_chars": len(prompt),
                    "followups": list(followups),
                    "turns": 1 + len(followups),
                    "timeout": {"turn": turn_timeout, "total": total_timeout},
                    "env": env_masked,
                }
                slots_data.append(slot)

                if not as_json:
                    env_summary = [f"    {k} = {v}" for k, v in env_masked.items()]
                    print(f"\n[{task.id}/{model_name}]  RUN")
                    print(f"  cwd: {run_dir}")
                    print(f"  cmd: {' '.join(cmd)}")
                    print(f"  stdin: <<< prompt  (piped, {len(prompt)} chars)")
                    print(f"  prompt (turn 1): {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
                    for i, fu in enumerate(followups, start=2):
                        print(f"  followup (turn {i}): {fu[:100]}{'...' if len(fu) > 100 else ''}")
                    print(f"  turns: 1 initial + {len(followups)} follow-up(s) = {1 + len(followups)} total")
                    print(f"  timeout: turn={turn_timeout}s  total={total_timeout}s")
                    print(f"  env:")
                    print("\n".join(env_summary))

        if as_json:
            return {
                "tasks": slots_data,
                "summary": {
                    "total_slots": len(tasks) * len(models),
                    "to_run": will_run,
                    "skipped": will_skip,
                },
            }

        print(f"\nTotal: {len(tasks)} task(s) x {len(models)} model(s) = "
              f"{len(tasks) * len(models)} slot(s): "
              f"{will_run} to run, {will_skip} already done")
        return

    state = PipelineState(config.state_path)
    tasks = select_delivery_tasks(config, task_ids)
    models = models or ["qwen", "claude"]
    # max_parallel = number of tasks running concurrently; each task runs 2 models
    sem = asyncio.Semaphore(config.max_parallel * 2)
    total_tasks = len(tasks)

    async def run_model_bounded(task: TaskConfig, model_name: str) -> None:
        async with sem:
            await run_task_model(task, config, state, model_name, turn_timeout, total_timeout)

    async def run_task_models(task: TaskConfig, index: int) -> None:
        print(f"[{task.id}] Starting run ({index}/{total_tasks}, {', '.join(models)})...")
        coros = [run_model_bounded(task, m) for m in models]
        results = await asyncio.gather(*coros, return_exceptions=True)
        for model_name, result in zip(models, results):
            if isinstance(result, Exception):
                print(f"[{task.id}/{model_name}] ERROR: {str(result)[:200]}")
                state.set(task.id, "run", model=model_name, status="failed", error=str(result)[:200])

    task_coros = [run_task_models(task, i + 1) for i, task in enumerate(tasks)]
    if task_coros:
        print(f"Pipeline: {total_tasks} tasks, max {config.max_parallel} concurrent ({config.max_parallel * 2} processes)")
        results = await asyncio.gather(*task_coros, return_exceptions=True)
        for result in results:
            if isinstance(result, Exception):
                print(f"  ERROR: {str(result)[:200]}")
    print("Run complete.")
