"""Regression tests for validate.py --tasks submission-ID lookup.

Verifies that validate(config, task_ids=[...]) uses full-list submission
IDs (computed from all tasks, not just the filtered subset) when looking
up rows in submission.csv.
"""

from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import PropertyMock, patch

from ctpipe.config import (
    REFERENCE_CRITERION_DESCRIPTIONS,
    SUBMISSION_FIELDNAMES,
    REFERENCE_CRITERION_NAMES,
    BatchConfig,
    ModelConfig,
    TaskConfig,
    write_task_manifest,
)
from ctpipe.finalize import finalize
from ctpipe.toml_utils import Criterion, write_quality_toml
from ctpipe.validate import validate


def _build_config(tasks: list[TaskConfig] | None = None) -> BatchConfig:
    return BatchConfig(
        delivery_date="20990101",
        runs_root=Path("D:/runs"),
        max_parallel=2,
        tasks=tasks or [],
        qwen=ModelConfig(auth_token="", base_url="", model="qwen-test"),
        claude=ModelConfig(auth_token="", base_url="", model="claude-test"),
        person_id="99",
    )


def _make_task(task_id: str = "CT-0001", task_type: str = "bug-fix") -> TaskConfig:
    return TaskConfig(
        id=task_id,
        project_path=Path("D:/projects/demo"),
        clone_method="git",
        task_type=task_type,
        domain="web_frontend",
        language="ts",
        prompt_qwen="qwen prompt",
        prompt_claude="claude prompt",
    )


def _write_trajectory(
    jsonl_path: Path, session_id: str, model_name: str, user_turns: int = 3
) -> None:
    """Write a minimal valid trajectory JSONL."""
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps({"sessionId": session_id, "type": "system"})]
    for i in range(user_turns):
        lines.append(
            json.dumps({"type": "user", "message": {"role": "user", "content": f"msg {i}"}})
        )
        lines.append(
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "role": "assistant",
                        "model": f"{model_name}-model-v1",
                        "content": f"reply {i}",
                    },
                }
            )
        )
    while len(lines) < 12:
        lines.append(json.dumps({"type": "system", "info": "padding"}))
    jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_score(score_path: Path, score_per_criterion: int = 3) -> None:
    """Write a complete score TOML with valid criterion names."""
    score_path.parent.mkdir(parents=True, exist_ok=True)
    names = REFERENCE_CRITERION_NAMES[:7]
    criteria = [
        Criterion(
            name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
            1.0, score_per_criterion, "评分理由"
        )
        for name in names
    ]
    write_quality_toml(score_path, criteria)


def _setup_delivery_for_validate(
    config: BatchConfig,
    tasks: list[TaskConfig],
    qwen_score: int = 2,
    claude_score: int = 5,
) -> None:
    """Create a complete delivery directory suitable for validate().

    Trajectories, scores, metadata, and submission CSV (via finalize)
    are all set up so that validate() passes cleanly.
    """
    delivery_dir = config.delivery_dir
    delivery_dir.mkdir(parents=True, exist_ok=True)
    write_task_manifest(config.task_manifest_path, tasks)

    for task in tasks:
        for model_name, session, score in [
            ("qwen", f"qw-{task.id}", qwen_score),
            ("claude", f"cl-{task.id}", claude_score),
        ]:
            _write_trajectory(
                delivery_dir / "trajectories" / model_name / f"{task.id}.jsonl",
                session, model_name,
            )
            _write_score(
                delivery_dir / "scores" / model_name / f"{task.id}.quality.toml",
                score,
            )

        meta_dir = delivery_dir / "metadata"
        meta_dir.mkdir(parents=True, exist_ok=True)
        (meta_dir / f"{task.id}.md").write_text(
            f"# Task {task.id}\n", encoding="utf-8"
        )


def _run_validate(config, task_ids=None) -> tuple[bool, str]:
    """Run validate() and capture stdout."""
    buf = io.StringIO()
    with redirect_stdout(buf):
        result = validate(config, task_ids=task_ids)
    return result, buf.getvalue()


# =========================================================================
# Regression: validate --tasks must use full-list submission IDs to find
# the correct CSV row.
# =========================================================================


class ValidatePartialTasksCsvLookupTest(unittest.TestCase):
    """Regression: validate(config, task_ids=['CT-0002']) must look up the
    CSV row using the full-list submission ID (seq=02), not the filtered
    subset ID (seq=01)."""

    def test_partial_validate_finds_csv_row_by_full_list_id(self) -> None:
        """3 bug-fix tasks finalized → validate --tasks CT-0002 must find
        '99-101-bug-fix-02' in submission.csv without reporting it missing."""
        tasks = [_make_task(f"CT-{i:04d}") for i in range(1, 4)]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = _build_config(tasks)
                _setup_delivery_for_validate(config, tasks)

                # Full finalize generates submission.csv with all 3 rows
                with redirect_stdout(io.StringIO()):
                    finalize(config)

                # Validate only CT-0002 — must find its row (99-101-bug-fix-02)
                result, output = _run_validate(config, task_ids=["CT-0002"])

        self.assertTrue(
            result,
            f"validate --tasks should pass but failed:\n{output}",
        )
        self.assertNotIn("submission row missing", output)

    def test_partial_validate_third_task_finds_its_own_id(self) -> None:
        """Validate --tasks CT-0003 must find seq=03, not seq=01."""
        tasks = [_make_task(f"CT-{i:04d}") for i in range(1, 4)]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = _build_config(tasks)
                _setup_delivery_for_validate(config, tasks)

                with redirect_stdout(io.StringIO()):
                    finalize(config)

                result, output = _run_validate(config, task_ids=["CT-0003"])

        self.assertTrue(
            result,
            f"validate --tasks CT-0003 should pass but failed:\n{output}",
        )
        self.assertNotIn("submission row missing", output)

    def test_partial_validate_mixed_types_keeps_correct_seq(self) -> None:
        """Two task types: bug-fix ×2 + feature ×1. Validating the second
        bug-fix must find seq=02 within bug-fix, not seq=01."""
        tasks = [
            _make_task("CT-0001", task_type="bug-fix"),
            _make_task("CT-0002", task_type="feature"),
            _make_task("CT-0003", task_type="bug-fix"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = _build_config(tasks)
                _setup_delivery_for_validate(config, tasks)

                with redirect_stdout(io.StringIO()):
                    finalize(config)

                # CT-0003 is the 2nd bug-fix → should be "99-101-bug-fix-02"
                result, output = _run_validate(config, task_ids=["CT-0003"])

        self.assertTrue(
            result,
            f"validate --tasks CT-0003 (2nd bug-fix) should pass:\n{output}",
        )
        self.assertNotIn("submission row missing", output)


if __name__ == "__main__":
    unittest.main()
