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
    p_run.add_argument("--turn-timeout", type=int, default=900, help="Timeout per turn in seconds")
    p_run.add_argument("--total-timeout", type=int, default=3600, help="Total timeout per task and model in seconds")

    p_collect = sub.add_parser("collect", help="Find and copy trajectory JSONL files")
    p_collect.add_argument("--tasks", nargs="*")
    p_collect.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])
    p_collect.add_argument("--no-salvage", action="store_true", help="Skip salvaging from interrupted runs")
    p_collect.add_argument("--force", action="store_true", help="Skip start_time/session_id validation, pick newest trajectory by mtime")

    p_score = sub.add_parser("score", help="AI-generate initial quality scores")
    p_score.add_argument("--tasks", nargs="*")
    p_score.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])
    p_score.add_argument("--auto-rescore", action="store_true",
                         help="Auto-trigger rescore for tasks failing passrate thresholds")

    p_finalize = sub.add_parser("finalize", help="Calculate passrates and generate submission.csv")
    p_finalize.add_argument("--tasks", nargs="*")
    p_finalize.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])

    p_validate = sub.add_parser("validate", help="Validate delivery files and submission consistency")
    p_validate.add_argument("--tasks", nargs="*")
    p_validate.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])

    p_gen = sub.add_parser("gen", help="Auto-generate tasks from GitHub/Gitee projects")
    p_gen.add_argument("--count", type=int, required=True, help="Number of tasks to generate")
    p_gen.add_argument("--domain", help="Filter by application domain")
    p_gen.add_argument("--language", help="Filter by programming language")
    p_gen.add_argument("--task-type", help="Filter by task type (e.g. bug-fix, feature)")
    p_gen.add_argument("--from-local", help="Use a local project path instead of searching GitHub")
    p_gen.add_argument("--clone-dir", help="Directory to clone projects into (default: runs_root)")
    p_gen.add_argument("--dry-run", action="store_true", help="Preview without cloning or writing")
    p_gen.add_argument("--clone-only", action="store_true", help="Only search and clone repos (skip AI task generation)")
    p_gen.add_argument("--analyze", action="store_true", help="Use Claude Code to analyze a local project and write task entries directly")
    p_gen.add_argument("--gen-timeout", type=int, default=900, help="Total timeout per task generation in seconds (default: 900)")
    p_gen.add_argument("--per-project", type=int, default=1, help="Generate N tasks per project (default: 1). Use 3-4 to save time.")
    p_gen.add_argument("--source", choices=["github", "gitee"], default="github", help="Project search source (default: github)")

    p_all = sub.add_parser("all", help="Run prepare -> run -> collect -> score -> finalize -> validate")
    p_all.add_argument("--tasks", nargs="*")
    p_all.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])
    p_all.add_argument("--turn-timeout", type=int, default=900)
    p_all.add_argument("--total-timeout", type=int, default=3600)
    p_all.add_argument("--auto-rescore", action="store_true", default=True,
                       help="Auto-trigger rescore for tasks failing passrate thresholds (default: on)")
    p_all.add_argument("--no-auto-rescore", action="store_false", dest="auto_rescore",
                       help="Disable auto-rescore after scoring")

    p_reset = sub.add_parser("reset", help="Reset pipeline state for specific tasks/stages to allow re-runs")
    p_reset.add_argument("--tasks", nargs="+", required=True, help="Task IDs to reset")
    p_reset.add_argument("--stages", nargs="+", required=True,
                         choices=["prepare", "run", "collect", "score", "finalize"],
                         help="Pipeline stages to reset")
    p_reset.add_argument("--models", nargs="*", choices=["qwen", "claude"],
                         help="Reset only specific models (default: both)")

    p_check = sub.add_parser("check", help="Deep validation: turns, models, sessions, scores, thresholds")
    p_check.add_argument("--tasks", nargs="*")
    p_check.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])

    p_rescore = sub.add_parser("rescore", help="Re-score with customized dimensions and descriptions")
    p_rescore.add_argument("--tasks", nargs="*", help="Specific task IDs to rescore")
    p_rescore.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])

    p_stats = sub.add_parser("stats", help="Show pipeline stage statistics")
    p_stats.add_argument("--tasks", nargs="*")
    p_stats.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])
    p_stats.add_argument("--format", choices=["table", "json"], default="table", dest="fmt")
    p_stats.add_argument("--timing", action="store_true", default=False, help="Include run/score duration statistics")

    p_retry = sub.add_parser("retry", help="Auto-retry failed/partial pipeline tasks")
    p_retry.add_argument("--tasks", nargs="*", help="Specific task IDs to retry")
    p_retry.add_argument("--stages", nargs="*",
                         choices=["prepare", "run", "collect", "score", "finalize"],
                         help="Only retry specific stages")
    p_retry.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])
    p_retry.add_argument("--max-retries", type=int, default=2, help="Max retry attempts per entry (default: 2)")
    p_retry.add_argument("--turn-timeout", type=int, default=900, help="Timeout per turn in seconds")
    p_retry.add_argument("--total-timeout", type=int, default=3600, help="Total timeout per task and model")
    p_retry.add_argument("--dry-run", action="store_true", help="Preview what would be retried")
    p_retry.add_argument("--no-cascade", action="store_true", help="Don't cascade retries to downstream stages")

    p_clean = sub.add_parser("clean", help="Post-delivery cleanup of runs, cache, and old deliveries")
    p_clean.add_argument("--tasks", nargs="*", help="Only clean specific task IDs")
    p_clean.add_argument("--no-runs", action="store_true", help="Skip cleaning runs/ directories")
    p_clean.add_argument("--cache", action="store_true", help="Also clean ~/.claude/projects/ JSONL cache")
    p_clean.add_argument("--old-deliveries", action="store_true", help="Also remove old delivery_* directories")
    p_clean.add_argument("--dry-run", action="store_true", help="Preview what would be deleted")

    p_export = sub.add_parser("export", help="Export delivery results as a structured JSON report")
    p_export.add_argument("--tasks", nargs="*", help="Specific task IDs to include")
    p_export.add_argument("--models", nargs="*", choices=["qwen", "claude"], default=["qwen", "claude"])
    p_export.add_argument("--output", help="Output file path (default: stdout)")

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
        result = collect_all(config, args.tasks, args.models, salvage=not args.no_salvage, force=args.force)

    elif args.command == "score":
        from ctpipe.score import score_all
        result = asyncio.run(score_all(config, args.tasks, args.models,
                                       auto_rescore=args.auto_rescore))

    elif args.command == "finalize":
        from ctpipe.finalize import finalize
        result = finalize(config, args.tasks, args.models)

    elif args.command == "validate":
        from ctpipe.validate import validate
        result = validate(config, args.tasks, args.models)

    elif args.command == "gen":
        from ctpipe.gen import generate
        from ctpipe.config import _validate_runs_root
        clone_dir_path = None
        if args.clone_dir:
            clone_dir_path = _validate_runs_root(Path(args.clone_dir))
        result = asyncio.run(generate(
            config,
            count=args.count,
            domain=args.domain,
            language=args.language,
            task_type=args.task_type,
            from_local=args.from_local,
            clone_dir=clone_dir_path,
            dry_run=args.dry_run,
            clone_only=args.clone_only,
            analyze=args.analyze,
            total_timeout=args.gen_timeout,
            per_project=args.per_project,
            source=args.source,
        ))

    elif args.command == "all":
        result = _run_all_stages(config, args)

    elif args.command == "check":
        from ctpipe.check import check
        result = check(config, args.tasks, args.models)

    elif args.command == "rescore":
        from ctpipe.rescore import rescore_all
        result = asyncio.run(rescore_all(config, args.tasks, args.models))

    elif args.command == "retry":
        from ctpipe.retry import retry
        result = asyncio.run(retry(
            config,
            task_ids=args.tasks,
            stages=args.stages,
            models=args.models,
            max_retries=args.max_retries,
            turn_timeout=args.turn_timeout,
            total_timeout=args.total_timeout,
            dry_run=args.dry_run,
            cascade=not args.no_cascade,
        ))

    elif args.command == "stats":
        from ctpipe.stats import show_stats
        result = show_stats(config, args.tasks, args.models, args.fmt, args.timing)

    elif args.command == "clean":
        from ctpipe.clean import clean
        clean(
            config,
            task_ids=args.tasks,
            runs=not args.no_runs,
            cache=args.cache,
            old_deliveries=args.old_deliveries,
            dry_run=args.dry_run,
        )
        result = 0

    elif args.command == "export":
        from ctpipe.export import export_report
        export_report(
            config,
            task_ids=args.tasks,
            models=args.models,
            output=Path(args.output) if args.output else None,
        )
        result = 0

    elif args.command == "reset":
        from ctpipe.state import PipelineState
        state = PipelineState(config.state_path)
        models = args.models or ["qwen", "claude"]
        count = 0
        for task_id in args.tasks:
            for stage in args.stages:
                if stage in MODEL_SPECIFIC_STAGES:
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
    from ctpipe.config import MODEL_SPECIFIC_STAGES, select_delivery_tasks
    from ctpipe.finalize import finalize
    from ctpipe.prepare import prepare
    from ctpipe.run import run_all
    from ctpipe.score import score_all
    from ctpipe.state import PipelineState
    from ctpipe.validate import validate

    def _check_stage(state: PipelineState, stage: str, models: list[str]) -> int:
        tasks = select_delivery_tasks(config, args.tasks)
        failed = 0
        partial = 0
        missing = 0
        for task in tasks:
            if stage in MODEL_SPECIFIC_STAGES:
                for m in models:
                    info = state.get(task.id, stage, m)
                    status = info.get("status", "")
                    if status == "failed":
                        failed += 1
                    elif status == "partial":
                        partial += 1
                    elif status in ("", "draft"):
                        missing += 1
            else:
                info = state.get(task.id, stage)
                status = info.get("status", "")
                if status == "failed":
                    failed += 1
                elif status == "partial":
                    partial += 1
                elif status in ("", "draft"):
                    missing += 1
        if failed:
            print(f"\n  WARNING: {failed} task(s) failed in {stage} stage")
        if partial:
            print(f"\n  WARNING: {partial} task(s) partially completed in {stage} stage")
        if missing:
            print(f"\n  WARNING: {missing} task(s) missing or incomplete in {stage} stage")
        return failed + partial + missing

    async def _async_pipeline() -> None:
        state = PipelineState(config.state_path)

        print("=" * 60)
        print("Stage 1/6: PREPARE")
        print("=" * 60)
        prepare(config, args.tasks)

        print("\n" + "=" * 60)
        print("Stage 2/6: RUN")
        print("=" * 60)
        await run_all(config, args.tasks, args.models, args.turn_timeout, args.total_timeout)
        state.reload()
        _check_stage(state, "run", args.models)

        print("\n" + "=" * 60)
        print("Stage 3/6: COLLECT")
        print("=" * 60)
        collect_all(config, args.tasks, args.models)
        state.reload()
        _check_stage(state, "collect", args.models)

        print("\n" + "=" * 60)
        print("Stage 4/6: SCORE")
        print("=" * 60)
        await score_all(config, args.tasks, args.models,
                        auto_rescore=args.auto_rescore)
        state.reload()
        _check_stage(state, "score", args.models)

        print("\n" + "=" * 60)
        print("Stage 5/6: FINALIZE")
        print("=" * 60)
        finalize(config, args.tasks, args.models)

    pipeline_error: Exception | None = None
    try:
        asyncio.run(_async_pipeline())
    except Exception as exc:
        pipeline_error = exc
        print(f"\nERROR in pipeline: {exc}")

    print("\n" + "=" * 60)
    print("Stage 6/6: VALIDATE")
    print("=" * 60)
    valid = validate(config, args.tasks, args.models)

    if pipeline_error:
        print(f"\nWARNING: pipeline had errors before validate: {pipeline_error}")
        return False
    return valid
