"""CLI entry point for ctpipe."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="ctpipe",
        description="Coding Trajectory automated pipeline",
    )
    parser.add_argument("--config", default="tasks.toml", help="Path to tasks.toml")
    parser.add_argument("--env", default=".env", help="Path to .env file")

    sub = parser.add_subparsers(dest="command", required=True)

    p_prepare = sub.add_parser("prepare", help="Clone projects and set up the delivery directory")
    p_prepare.add_argument("--tasks", nargs="*", help="Specific task IDs")

    p_run = sub.add_parser("run", help="Execute claude -p for each task and model")
    p_run.add_argument("--tasks", nargs="*", help="Specific task IDs")
    p_run.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])
    p_run.add_argument("--turn-timeout", type=int, default=600, help="Timeout per turn in seconds")
    p_run.add_argument("--total-timeout", type=int, default=1800, help="Total timeout per task and model in seconds")

    p_collect = sub.add_parser("collect", help="Find and copy trajectory JSONL files")
    p_collect.add_argument("--tasks", nargs="*")

    p_score = sub.add_parser("score", help="AI-generate initial quality scores")
    p_score.add_argument("--tasks", nargs="*")
    p_score.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])

    p_finalize = sub.add_parser("finalize", help="Calculate passrates and generate submission.csv")
    p_finalize.add_argument("--tasks", nargs="*")

    p_validate = sub.add_parser("validate", help="Validate delivery files and submission consistency")
    p_validate.add_argument("--tasks", nargs="*")

    p_gen = sub.add_parser("gen", help="Auto-generate tasks from GitHub projects")
    p_gen.add_argument("--count", type=int, required=True, help="Number of tasks to generate")
    p_gen.add_argument("--domain", help="Filter by application domain")
    p_gen.add_argument("--language", help="Filter by programming language")
    p_gen.add_argument("--task-type", help="Filter by task type (e.g. bug-fix, feature)")
    p_gen.add_argument("--from-local", help="Use a local project path instead of searching GitHub")
    p_gen.add_argument("--clone-dir", help="Directory to clone projects into (default: runs_root)")
    p_gen.add_argument("--dry-run", action="store_true", help="Preview without cloning or writing")
    p_gen.add_argument("--gen-timeout", type=int, default=900, help="Total timeout per task generation in seconds (default: 900)")
    p_gen.add_argument("--per-project", type=int, default=1, help="Generate N tasks per project (default: 1). Use 3-4 to save time.")

    p_all = sub.add_parser("all", help="Run prepare -> run -> collect -> score -> finalize -> validate")
    p_all.add_argument("--tasks", nargs="*")
    p_all.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])
    p_all.add_argument("--turn-timeout", type=int, default=600)
    p_all.add_argument("--total-timeout", type=int, default=1800)

    p_reset = sub.add_parser("reset", help="Reset pipeline state for specific tasks/stages to allow re-runs")
    p_reset.add_argument("--tasks", nargs="+", required=True, help="Task IDs to reset")
    p_reset.add_argument("--stages", nargs="+", required=True,
                         choices=["prepare", "run", "collect", "score", "finalize"],
                         help="Pipeline stages to reset")
    p_reset.add_argument("--models", nargs="*", choices=["qwen", "claude"],
                         help="Reset only specific models (default: both)")

    args = parser.parse_args()
    base_dir = Path(__file__).resolve().parent.parent
    config_path = base_dir / args.config
    env_path = base_dir / args.env

    from ctpipe.config import load_config

    config = load_config(config_path, env_path)
    result: int | bool | None = None

    if args.command == "prepare":
        from ctpipe.prepare import prepare
        result = prepare(config, args.tasks)

    elif args.command == "run":
        from ctpipe.run import run_all
        result = asyncio.run(run_all(config, args.tasks, args.models, args.turn_timeout, args.total_timeout))

    elif args.command == "collect":
        from ctpipe.collect import collect_all
        result = collect_all(config, args.tasks)

    elif args.command == "score":
        from ctpipe.score import score_all
        result = asyncio.run(score_all(config, args.tasks, args.models))

    elif args.command == "finalize":
        from ctpipe.finalize import finalize
        result = finalize(config, args.tasks)

    elif args.command == "validate":
        from ctpipe.validate import validate
        result = validate(config, args.tasks)

    elif args.command == "gen":
        from ctpipe.gen import generate
        result = asyncio.run(generate(
            config,
            count=args.count,
            domain=args.domain,
            language=args.language,
            task_type=args.task_type,
            from_local=args.from_local,
            clone_dir=Path(args.clone_dir) if args.clone_dir else None,
            dry_run=args.dry_run,
            total_timeout=args.gen_timeout,
            per_project=args.per_project,
        ))

    elif args.command == "all":
        result = _run_all_stages(config, args)

    elif args.command == "reset":
        from ctpipe.state import PipelineState
        state = PipelineState(config.delivery_dir / "pipeline_state.json")
        models = args.models or ["qwen", "claude"]
        count = 0
        for task_id in args.tasks:
            for stage in args.stages:
                if stage in ("run", "collect", "score"):
                    for model in models:
                        if state.reset(task_id, stage, model):
                            print(f"  Reset {task_id}/{stage}/{model}")
                            count += 1
                else:
                    if state.reset(task_id, stage):
                        print(f"  Reset {task_id}/{stage}")
                        count += 1
        print(f"Reset {count} state entries.")
        result = 0

    if isinstance(result, bool):
        return 0 if result else 1
    if isinstance(result, int):
        return result
    return 0


def _run_all_stages(config, args) -> bool:
    from ctpipe.collect import collect_all
    from ctpipe.finalize import finalize
    from ctpipe.prepare import prepare
    from ctpipe.run import run_all
    from ctpipe.score import score_all
    from ctpipe.validate import validate

    async def _async_pipeline() -> None:
        print("=" * 60)
        print("Stage 1/6: PREPARE")
        print("=" * 60)
        prepare(config, args.tasks)

        print("\n" + "=" * 60)
        print("Stage 2/6: RUN")
        print("=" * 60)
        await run_all(config, args.tasks, args.models, args.turn_timeout, args.total_timeout)

        print("\n" + "=" * 60)
        print("Stage 3/6: COLLECT")
        print("=" * 60)
        collect_all(config, args.tasks)

        print("\n" + "=" * 60)
        print("Stage 4/6: SCORE")
        print("=" * 60)
        await score_all(config, args.tasks, args.models)

        print("\n" + "=" * 60)
        print("Stage 5/6: FINALIZE")
        print("=" * 60)
        finalize(config, args.tasks)

    asyncio.run(_async_pipeline())

    print("\n" + "=" * 60)
    print("Stage 6/6: VALIDATE")
    print("=" * 60)
    return validate(config, args.tasks)
