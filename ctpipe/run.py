"""run subcommand: execute claude -p with multi-turn follow-ups via --resume."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from ctpipe.config import BatchConfig, ModelConfig, TaskConfig, build_claude_env, select_tasks
from ctpipe.state import PipelineState


@dataclass
class TurnResult:
    turn: int
    exit_code: int
    stdout: str
    stderr: str
    duration_s: float
    session_id: str = ""


def _build_env(model_config: ModelConfig, http_proxy: str = "") -> dict[str, str]:
    missing = []
    if not model_config.auth_token:
        missing.append("auth token")
    if not model_config.base_url:
        missing.append("base URL")
    if not model_config.model:
        missing.append("model")
    if missing:
        raise ValueError(f"Model config is incomplete: missing {', '.join(missing)}")
    return build_claude_env(model_config, http_proxy=http_proxy)


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
    timeout: int = 600,
) -> TurnResult:
    cmd = [
        "claude",
        "-p",
        prompt,
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
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return TurnResult(
            turn=0,
            exit_code=-1,
            stdout="",
            stderr="TIMEOUT",
            duration_s=time.time() - started_at,
        )

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
    turn_timeout: int = 600,
    total_timeout: int = 1800,
    http_proxy: str = "",
) -> dict[str, object]:
    env = _build_env(model_config, http_proxy=http_proxy)
    total_start = time.time()
    start_time = time.time()
    turns_completed = 0
    session_id = ""
    had_errors = False
    all_results: list[dict[str, object]] = []

    print(f"  [{task.id}/{model_name}] Turn 1/{1 + len(followups)}: initial prompt...")
    result = await _run_claude_p(prompt, env, run_dir, model=model_config.model, timeout=turn_timeout)
    result.turn = 1
    turns_completed = 1
    session_id = result.session_id
    had_errors = result.exit_code != 0

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
            "duration_s": round(time.time() - total_start, 1),
            "start_time": start_time,
            "turns_detail": all_results,
            "error": error,
        }
        state.set(task.id, "run", model=model_name, **summary)
        return summary

    if result.exit_code != 0:
        print(f"  [{task.id}/{model_name}] Turn 1 exited with {result.exit_code}; continuing with captured session")

    for index, followup in enumerate(followups, start=2):
        elapsed = time.time() - total_start
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

        if result.session_id and not session_id:
            session_id = result.session_id

        all_results.append({
            "turn": index,
            "exit_code": result.exit_code,
            "duration_s": round(result.duration_s, 1),
        })

        if result.exit_code != 0:
            print(f"  [{task.id}/{model_name}] Turn {index} failed (exit={result.exit_code}), continuing...")

    total_duration = time.time() - total_start
    summary = {
        "status": "done",
        "session_id": session_id,
        "turns": turns_completed,
        "duration_s": round(total_duration, 1),
        "start_time": start_time,
        "turns_detail": all_results,
        "had_errors": had_errors,
    }
    state.set(task.id, "run", model=model_name, **summary)
    print(f"  [{task.id}/{model_name}] Done: {turns_completed} turns in {total_duration:.0f}s")
    return summary


async def run_task_pair(
    task: TaskConfig,
    config: BatchConfig,
    state: PipelineState,
    turn_timeout: int = 600,
    total_timeout: int = 1800,
    models: list[str] | None = None,
) -> None:
    models = models or ["qwen", "claude"]
    coroutines = []
    prepare_info = state.get(task.id, "prepare")

    for model_name in models:
        if state.is_done(task.id, "run", model_name):
            print(f"[{task.id}/{model_name}] run already done, skipping")
            continue

        run_dir = Path(prepare_info.get(f"{model_name}_dir", ""))
        if not run_dir.is_dir():
            print(f"[{task.id}/{model_name}] ERROR: run dir not found: {run_dir}")
            state.set(task.id, "run", model=model_name, status="failed", error="run dir not found")
            continue

        model_config = config.qwen if model_name == "qwen" else config.claude
        prompt = task.prompt_qwen if model_name == "qwen" else task.prompt_claude
        followups = task.followups_qwen if model_name == "qwen" else task.followups_claude

        coroutines.append(
            run_single(
                task,
                model_name,
                model_config,
                run_dir,
                prompt,
                followups,
                state,
                turn_timeout,
                total_timeout,
                http_proxy=config.http_proxy,
            )
        )

    if coroutines:
        await asyncio.gather(*coroutines, return_exceptions=True)


async def run_all(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
    turn_timeout: int = 600,
    total_timeout: int = 1800,
) -> None:
    state = PipelineState(config.delivery_dir / "pipeline_state.json")
    tasks = select_tasks(config.tasks, task_ids)
    sem = asyncio.Semaphore(config.max_parallel)

    async def bounded(task: TaskConfig) -> None:
        async with sem:
            print(f"[{task.id}] Starting run...")
            await run_task_pair(task, config, state, turn_timeout, total_timeout, models)

    await asyncio.gather(*(bounded(task) for task in tasks), return_exceptions=True)
    print("Run complete.")
