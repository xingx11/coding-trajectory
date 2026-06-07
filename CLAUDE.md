# CLAUDE.md

This file provides repository guidance for Claude Code and other coding agents working in this workspace.

## What This Repository Is

This is a data-production workspace, not a normal application repository. It produces Coding Trajectory benchmark samples that compare Qwen and Claude on real local coding tasks.

Each sample row consists of:

- one Qwen trajectory JSONL
- one Claude trajectory JSONL
- one Qwen rubric score file
- one Claude rubric score file
- metadata describing the task and sessions
- a submission row with passrates and task labels

## Passrate Thresholds

Every submitted row should satisfy:

```text
qwen passrate  < 0.7
claude passrate >= 0.71
claude passrate > qwen passrate
(claude passrate - qwen passrate) / qwen passrate > 20%
```

Passrate formula:

```text
sum(score_i × weight_i) / sum(points_i × weight_i)
```

With 7 criteria × 5 points × weight 1.0, this simplifies to `sum(7 scores) / 35`.

## Key Commands

Calculate passrates:

```powershell
python rubrics_templates\calc_passrate.py <dir_or_file> [<dir_or_file> ...]
```

Run the normalized automation pipeline:

```powershell
python -m ctpipe --config tasks.toml --env .env all
```

Clone repos only (skip AI analysis):

```powershell
python -m ctpipe gen --clone-only --count 6 --per-project 3
```

Analyze a local repo and write task entries:

```powershell
python -m ctpipe gen --from-local "<path>" --count 3 --analyze
```

Validate the current delivery batch:

```powershell
python -m ctpipe validate
```

## Directory Layout

- `docs/`
  - reference docs, templates, task bank, examples, and supporting assets
- `docs/examples/`
  - reference JSONL trajectories and scored TOML files
- `docs/assets/`
  - supporting screenshots and clarification images
- `rubrics_templates/`
  - blank scoring templates and passrate tooling
- `delivery_YYYYMMDD/`
  - a delivery batch containing trajectories, scores, metadata, tools, and submission files
- `ctpipe/`
  - the automation pipeline

## Delivery Workflow

Per-task rhythm:

1. Clone the project into isolated run directories.
2. Configure Qwen and Claude endpoint environment variables.
3. Run Claude Code against the chosen model endpoint.
4. Apply follow-up prompts as needed.
5. Collect the resulting trajectory JSONL.
6. Verify the model and session ID.
7. Fill the score sheets.
8. Finalize and validate the submission batch.

## Important Constraints

- Never commit tokens, `.env`, or any real API keys.
- Do not manually edit exported trajectory JSONL files.
- Session IDs must be read from the trajectory contents.
- Qwen and Claude may use different follow-ups, but they must stay on the same codebase and task theme for the same task ID.
- Python 3.11+ is required for `tomllib`.
- Claude Code scaffold version must be `@anthropic-ai/claude-code@2.1.86`.

## Scaffold Setup

```bash
bash scripts/setup_scaffold.sh
```

## Submission Fields

Each row in `submission.csv` must include a `命中QwenBad Pattern` value from:

| Pattern Key | Description |
|---|---|
| `lazy_shortcut` | 偷懒 — 模型只做核心功能，忽略隐式质量要求 |
| `poor_interaction` | 交互不通畅 — 模型不与用户确认就直接执行 |
| `github_based` | 基于GitHub的题目 — 需要web search定位旧版本bug |
| `environment_dependency` | 环境依赖 — venv/cuda/python版本等环境陷阱 |
| `instruction_follow` | 指令follow — 不遵循CLAUDE.md或项目约束 |
| `attachment_binary` | 附件处理不足 — 不主动处理PDF/zip/图片等附件 |
| `planning_only` | 只做准备/只写计划 — 不执行实质动作或不使用自定义工具 |
| `macos_development` | macOS开发能力不足 — 套用Linux方案 |
| `parallel_tool_usage` | 并行能力不足 — 串行处理可并行子任务 |

## Valid Task Types

`bug-fix`, `feature`, `enhancement`, `from_scratch`, `testing-quality`, `refactor-maintenance`, `build-release-config`, `documentation`, `code-explanation`, `security-compliance`
