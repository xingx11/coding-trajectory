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
```

Passrate formula:

```text
sum(7 scores) / 35
```

## Key Commands

Calculate passrates:

```powershell
python rubrics_templates\calc_passrate.py <dir_or_file> [<dir_or_file> ...]
```

Copy the most recent Claude Code trajectory into the delivery batch:

```powershell
powershell -ExecutionPolicy Bypass -File delivery_20260603\tools\copy_latest_claude_jsonl.ps1 -Model <qwen|claude> -TaskId <CT-xxxx>
```

Verify which model produced a trajectory file:

```powershell
powershell -ExecutionPolicy Bypass -File delivery_20260603\tools\inspect_jsonl_model.ps1 "<path_to_jsonl>"
```

Run the normalized automation pipeline:

```powershell
python -m ctpipe --config tasks.toml --env .env all
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
