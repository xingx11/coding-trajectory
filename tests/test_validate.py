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
    model_stem,
    write_task_manifest,
)
from ctpipe.finalize import finalize
from ctpipe.toml_utils import Criterion, write_quality_toml
from conftest import build_config, make_task, write_trajectory, write_score
from ctpipe.validate import validate


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
            write_trajectory(
                delivery_dir / "trajectories" / model_name / f"{model_stem(task.id, model_name)}.jsonl",
                session, model_name,
            )
            write_score(
                delivery_dir / "scores" / model_name / f"{model_stem(task.id, model_name)}.quality.toml",
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
        tasks = [make_task(f"CT-{i:04d}") for i in range(1, 4)]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config(tasks)
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
        tasks = [make_task(f"CT-{i:04d}") for i in range(1, 4)]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config(tasks)
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
            make_task("CT-0001", task_type="bug-fix"),
            make_task("CT-0002", task_type="feature"),
            make_task("CT-0003", task_type="bug-fix"),
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = build_config(tasks)
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
