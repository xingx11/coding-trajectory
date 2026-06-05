"""score subcommand: AI-generate initial quality scores from trajectories."""

from __future__ import annotations

import asyncio
import tomllib

from ctpipe import strip_claude_wrapper
from ctpipe.config import BatchConfig, TaskConfig, build_claude_env, select_delivery_tasks
from ctpipe.state import PipelineState
from ctpipe.toml_utils import Criterion, calc_passrate, read_quality_toml, write_quality_toml
from ctpipe.trajectory import extract_for_scoring

SCORING_SYSTEM_PROMPT = """You are reviewing a coding-assistant trajectory and filling a quality rubric.

You will receive:
1. A TOML scoring template.
2. A condensed trajectory transcript.

Instructions:
- Score each criterion from 0 to 5 as an integer.
- Write the rationale in Chinese.
- Keep `name`, `description`, `type`, `points`, and `weight` unchanged.
- Only modify `score` and `rationale`.
- Output TOML only. Do not wrap it in Markdown.

Scoring guidance:
- 5: complete, well verified, strong evidence
- 4: mostly complete with minor gaps
- 3: main path done but notable omissions remain
- 2: superficial implementation, important requirements missing
- 1: little meaningful progress
- 0: wrong direction, failed badly, or no attempt
"""


async def _call_scoring_ai(
    trajectory_text: str,
    template_text: str,
    env: dict[str, str],
    model: str = "",
    timeout: int = 300,
) -> str:
    user_prompt = (
        f"{SCORING_SYSTEM_PROMPT}\n\n"
        f"---\n\n"
        f"## Scoring Template\n\n{template_text}\n\n"
        f"## Trajectory To Score\n\n{trajectory_text}\n\n"
        "Return the completed TOML only."
    )

    cmd = [
        "claude",
        "-p",
        user_prompt,
        "--output-format",
        "text",
        "--dangerously-skip-permissions",
        "--setting-sources", "local",
        "--bare",
    ]
    if model:
        cmd += ["--model", model]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        return ""

    if proc.returncode != 0:
        err = stderr.decode("utf-8", errors="replace").strip()
        print(f"  WARNING: claude -p scoring exited with {proc.returncode}: {err[:300]}")

    return stdout.decode("utf-8", errors="replace")


def _parse_scored_toml(raw: str, template_criteria: list[Criterion]) -> list[Criterion] | None:
    cleaned = strip_claude_wrapper(raw)

    try:
        data = tomllib.loads(cleaned)
    except Exception:
        return None

    scored = data.get("criterion", [])
    if len(scored) != len(template_criteria):
        return None

    result: list[Criterion] = []
    for scored_item, template_item in zip(scored, template_criteria):
        score = int(scored_item.get("score", 0))
        if score < 0 or score > 5:
            return None
        rationale = scored_item.get("rationale", "")
        if not rationale:
            return None
        result.append(
            Criterion(
                name=template_item.name,
                description=template_item.description,
                type=template_item.type,
                points=template_item.points,
                weight=template_item.weight,
                score=score,
                rationale=rationale,
            )
        )
    return result


def _build_scoring_env(config: BatchConfig) -> dict[str, str]:
    if not config.claude.auth_token or not config.claude.base_url or not config.claude.model:
        raise ValueError("Claude scoring config is incomplete in .env (need auth token, base URL, and model)")
    return build_claude_env(config.claude)


async def score_single(
    task: TaskConfig,
    model_name: str,
    config: BatchConfig,
    state: PipelineState,
    env: dict[str, str],
) -> bool:
    collect_info = state.get(task.id, "collect", model_name)
    if collect_info.get("status") != "done":
        print(f"  [{task.id}/{model_name}] collect not done, skipping score")
        return False

    jsonl_rel = collect_info.get("jsonl_path", "")
    jsonl_path = config.delivery_dir / jsonl_rel
    if not jsonl_path.exists():
        print(f"  [{task.id}/{model_name}] ERROR: JSONL not found: {jsonl_path}")
        state.set(task.id, "score", model=model_name, status="failed", error="JSONL not found")
        return False

    template_path = config.delivery_dir / "scores" / model_name / f"{task.id}.quality.toml"
    if not template_path.exists():
        template_path = config.rubrics_dir / model_name / f"{task.id}.quality.toml"
    if not template_path.exists():
        print(f"  [{task.id}/{model_name}] ERROR: scoring template not found")
        state.set(task.id, "score", model=model_name, status="failed", error="template not found")
        return False

    template_criteria = read_quality_toml(template_path)
    template_text = template_path.read_text(encoding="utf-8")

    print(f"  [{task.id}/{model_name}] Extracting trajectory content...")
    trajectory_text = extract_for_scoring(jsonl_path)
    print(f"  [{task.id}/{model_name}] Trajectory: {len(trajectory_text)} chars")

    print(f"  [{task.id}/{model_name}] Calling AI for scoring...")
    raw_output = await _call_scoring_ai(trajectory_text, template_text, env, model=config.claude.model)

    if not raw_output:
        print(f"  [{task.id}/{model_name}] ERROR: empty AI response")
        state.set(task.id, "score", model=model_name, status="failed", error="empty response")
        return False

    scored = _parse_scored_toml(raw_output, template_criteria)
    output_path = config.delivery_dir / "scores" / model_name / f"{task.id}.quality.toml"

    if scored is None:
        draft_path = output_path.with_suffix(".draft.txt")
        draft_path.write_text(raw_output, encoding="utf-8")
        print(f"  [{task.id}/{model_name}] WARNING: could not parse AI output, saved to {draft_path.name}")
        state.set(task.id, "score", model=model_name, status="draft", draft_path=str(draft_path))
        return False

    write_quality_toml(output_path, scored)
    passrate = calc_passrate(scored)
    print(f"  [{task.id}/{model_name}] Scored: passrate={passrate:.4f}")
    state.set(task.id, "score", model=model_name, status="done", passrate=round(passrate, 4))
    return True


async def score_all(
    config: BatchConfig,
    task_ids: list[str] | None = None,
    models: list[str] | None = None,
) -> None:
    state = PipelineState(config.delivery_dir / "pipeline_state.json")
    models = models or ["qwen", "claude"]
    tasks = select_delivery_tasks(config, task_ids)
    sem = asyncio.Semaphore(config.max_parallel)

    async def bounded(task: TaskConfig, model_name: str) -> None:
        async with sem:
            await score_single(task, model_name, config, state, env)

    coros = []
    task_models: list[tuple[str, str]] = []
    for task in tasks:
        for model_name in models:
            if state.is_done(task.id, "score", model_name):
                print(f"[{task.id}/{model_name}] score already done, skipping")
                continue
            print(f"[{task.id}/{model_name}] Scoring trajectory...")
            coros.append(bounded(task, model_name))
            task_models.append((task.id, model_name))

    if coros:
        env = _build_scoring_env(config)
        results = await asyncio.gather(*coros, return_exceptions=True)
        for task_model, result in zip(task_models, results):
            if isinstance(result, Exception):
                task_id, model_name = task_model
                print(f"[{task_id}/{model_name}] ERROR: {result}")
                state.set(task_id, "score", model=model_name, status="failed", error=str(result))
    print("Score complete.")
