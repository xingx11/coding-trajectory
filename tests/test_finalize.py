from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from ctpipe.config import (
    SUBMISSION_FIELDNAMES,
    SUBMISSION_KEY_MAP,
    BatchConfig,
    ModelConfig,
    TaskConfig,
    write_task_manifest,
)
from ctpipe.finalize import _update_metadata_files, _write_submission_csv, finalize
from ctpipe.state import PipelineState
from ctpipe.toml_utils import Criterion, write_quality_toml


def _build_config(tasks: list[TaskConfig] | None = None) -> BatchConfig:
    return BatchConfig(
        delivery_date="20990101",
        runs_root=Path("D:/runs"),
        max_parallel=2,
        tasks=tasks or [],
        qwen=ModelConfig(auth_token="", base_url="", model="qwen-test"),
        claude=ModelConfig(auth_token="", base_url="", model="claude-test"),
    )


def _make_task(task_id: str = "CT-0001") -> TaskConfig:
    return TaskConfig(
        id=task_id,
        project_path=Path("D:/projects/demo"),
        clone_method="git",
        task_type="bug-fix",
        domain="web_frontend",
        language="ts",
        prompt_qwen="qwen prompt",
        prompt_claude="claude prompt",
        bad_pattern="lazy_shortcut",
    )


def _setup_task_scores(
    delivery_dir: Path,
    state: PipelineState,
    task_id: str,
    model_name: str,
    passrate: float,
    session_id: str = "s1",
) -> None:
    """Create trajectory file, score file, and state entries for one task/model."""
    traj_dir = delivery_dir / "trajectories" / model_name
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_file = traj_dir / f"{task_id}.jsonl"
    traj_file.write_text(
        json.dumps({"sessionId": session_id}) + "\n",
        encoding="utf-8",
    )

    score_dir = delivery_dir / "scores" / model_name
    score_dir.mkdir(parents=True, exist_ok=True)
    score_val = int(passrate * 100)
    criteria = [
        Criterion(f"c{i}", "desc", "likert", 100, 1.0, score_val, "ok")
        for i in range(7)
    ]
    write_quality_toml(score_dir / f"{task_id}.quality.toml", criteria)

    state.set(
        task_id, "collect", model=model_name,
        status="done", session_id=session_id, model_detected=model_name,
        jsonl_path=f"trajectories/{model_name}/{task_id}.jsonl",
    )
    state.set(
        task_id, "run", model=model_name,
        status="done", session_id=session_id, turns=3,
    )


# =========================================================================
# Multi-task CSV: 3 tasks finalized → CSV has header + 3 data rows,
# every SUBMISSION_FIELDNAMES column present.
# =========================================================================


class FinalizeMultiTaskCSVTest(unittest.TestCase):

    def test_three_tasks_produce_three_csv_rows_with_all_columns(self) -> None:
        tasks = [_make_task(f"CT-{i:04d}") for i in range(1, 4)]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config(tasks)
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, tasks)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                for task in tasks:
                    _setup_task_scores(delivery_dir, state, task.id, "qwen", 0.50, f"qw-{task.id}")
                    _setup_task_scores(delivery_dir, state, task.id, "claude", 0.85, f"cl-{task.id}")
                state.save()

                finalize(config)

                csv_path = delivery_dir / "submission.csv"
                self.assertTrue(csv_path.exists(), "submission.csv not created")

                with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                    reader = csv.DictReader(f)
                    fieldnames = reader.fieldnames
                    rows = list(reader)

        self.assertEqual(len(rows), 3, f"Expected 3 data rows, got {len(rows)}")
        for col in SUBMISSION_FIELDNAMES:
            self.assertIn(col, fieldnames, f"Missing column: {col}")
        task_ids = {row["id"] for row in rows}
        self.assertEqual(task_ids, {"CT-0001", "CT-0002", "CT-0003"})


# =========================================================================
# Dual-model thresholds
# =========================================================================


class FinalizeDualModelThresholdTest(unittest.TestCase):

    def _run_finalize_and_get_status(
        self,
        qwen_passrate: float,
        claude_passrate: float,
    ) -> str:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                _setup_task_scores(delivery_dir, state, task.id, "qwen", qwen_passrate, "qw-s1")
                _setup_task_scores(delivery_dir, state, task.id, "claude", claude_passrate, "cl-s1")
                state.save()

                finalize(config)

                state2 = PipelineState(delivery_dir / "pipeline_state.json")
                return state2.get(task.id, "finalize").get("status", "")

    def test_qwen_passrate_at_threshold_is_partial(self) -> None:
        # THRESHOLD_QWEN_MAX = 0.7; qwen=0.75 >= 0.7 → partial
        status = self._run_finalize_and_get_status(qwen_passrate=0.75, claude_passrate=0.95)
        self.assertEqual(status, "partial")

    def test_qwen_passrate_exactly_at_max_is_partial(self) -> None:
        # qwen=0.70 >= 0.7 → partial
        status = self._run_finalize_and_get_status(qwen_passrate=0.70, claude_passrate=0.95)
        self.assertEqual(status, "partial")

    def test_relative_gain_below_threshold_is_partial(self) -> None:
        # qwen=0.50, claude=0.55, gain=(0.55-0.50)/0.50=10% <= 20% → partial
        # Also claude <= qwen is false (0.55 > 0.50), so that check passes.
        # But gain <= 0.2 triggers threshold_ok=False → partial.
        status = self._run_finalize_and_get_status(qwen_passrate=0.50, claude_passrate=0.55)
        self.assertEqual(status, "partial")

    def test_claude_below_min_is_partial(self) -> None:
        # THRESHOLD_CLAUDE_MIN = 0.71; claude=0.65 < 0.71 → partial
        status = self._run_finalize_and_get_status(qwen_passrate=0.30, claude_passrate=0.65)
        self.assertEqual(status, "partial")

    def test_claude_leq_qwen_is_partial(self) -> None:
        # claude <= qwen → partial (even if both under thresholds)
        status = self._run_finalize_and_get_status(qwen_passrate=0.40, claude_passrate=0.40)
        self.assertEqual(status, "partial")

    def test_all_thresholds_met_is_done(self) -> None:
        # qwen=0.50 < 0.7, claude=0.85 >= 0.71, claude > qwen,
        # gain=(0.85-0.50)/0.50=70% > 20% → done
        status = self._run_finalize_and_get_status(qwen_passrate=0.50, claude_passrate=0.85)
        self.assertEqual(status, "done")


# =========================================================================
# _write_submission_csv field mapping
# =========================================================================


class WriteSubmissionCSVFieldMappingTest(unittest.TestCase):

    def test_all_key_map_entries_produce_correct_csv_values(self) -> None:
        row = {
            "id": "CT-9999",
            "qwen_trajectory": "trajectories/qwen/CT-9999.jsonl",
            "qwen_session_id": "qw-session-abc",
            "qwen_score_path": "scores/qwen/CT-9999.quality.toml",
            "claude_trajectory": "trajectories/claude/CT-9999.jsonl",
            "claude_session_id": "cl-session-xyz",
            "claude_score_path": "scores/claude/CT-9999.quality.toml",
            "qwen_passrate": "0.5000",
            "claude_passrate": "0.8500",
            "task_type": "bug-fix",
            "domain": "web_frontend",
            "language": "ts",
            "bad_pattern": "lazy_shortcut",
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "submission.csv"
            _write_submission_csv(csv_path, [row])

            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

        self.assertEqual(len(rows), 1)
        csv_row = rows[0]

        for csv_col, internal_key in SUBMISSION_KEY_MAP.items():
            self.assertIn(csv_col, csv_row, f"CSV missing column: {csv_col}")
            self.assertEqual(
                csv_row[csv_col],
                row[internal_key],
                f"Column '{csv_col}' should map to key '{internal_key}' "
                f"(expected '{row[internal_key]}', got '{csv_row[csv_col]}')",
            )

    def test_csv_fieldnames_match_submission_fieldnames(self) -> None:
        row = {"id": "CT-0001"}

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "submission.csv"
            _write_submission_csv(csv_path, [row])

            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])

        self.assertEqual(fieldnames, SUBMISSION_FIELDNAMES)

    def test_missing_internal_key_maps_to_empty_string(self) -> None:
        row = {"id": "CT-0001"}

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "submission.csv"
            _write_submission_csv(csv_path, [row])

            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                csv_row = next(reader)

        for csv_col in SUBMISSION_KEY_MAP:
            self.assertEqual(csv_row[csv_col], "", f"Missing key should produce empty string for '{csv_col}'")


# =========================================================================
# _update_metadata_files: regex back-fill must stay within its section;
# Qwen session_id must not leak into Claude section and vice versa.
# =========================================================================


METADATA_TEMPLATE = """\
# Task {task_id}

## Qwen Conversation
- Session id:
- Round count:

## Claude Conversation
- Session id:
- Round count:

- Qwen passrate:
- Claude passrate:
"""


class UpdateMetadataFilesRegexTest(unittest.TestCase):

    def _run_update(
        self,
        task_id: str = "CT-0001",
        qwen_session: str = "qw-sess-AAA",
        claude_session: str = "cl-sess-BBB",
        qwen_passrate: str = "0.5000",
        claude_passrate: str = "0.9000",
    ) -> str:
        task = _make_task(task_id)
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                metadata_dir = delivery_dir / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)

                md_path = metadata_dir / f"{task_id}.md"
                md_path.write_text(METADATA_TEMPLATE.format(task_id=task_id), encoding="utf-8")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task_id, "run", model="qwen", status="done", turns=5)
                state.set(task_id, "run", model="claude", status="done", turns=8)
                state.save()

                rows = [{
                    "id": task_id,
                    "qwen_session_id": qwen_session,
                    "claude_session_id": claude_session,
                    "qwen_passrate": qwen_passrate,
                    "claude_passrate": claude_passrate,
                }]

                _update_metadata_files(config, [task], rows, state)

                return md_path.read_text(encoding="utf-8")

    def test_qwen_session_id_not_in_claude_section(self) -> None:
        content = self._run_update(qwen_session="QWEN-ONLY-ID", claude_session="CLAUDE-ONLY-ID")
        lines = content.splitlines()
        in_claude = False
        for line in lines:
            if line.startswith("## Claude"):
                in_claude = True
            elif line.startswith("## ") and in_claude:
                in_claude = False
            if in_claude and "Session id:" in line:
                self.assertNotIn("QWEN-ONLY-ID", line,
                                 "Qwen session_id leaked into Claude section")
                self.assertIn("CLAUDE-ONLY-ID", line,
                              "Claude section should have Claude session_id")

    def test_claude_session_id_not_in_qwen_section(self) -> None:
        content = self._run_update(qwen_session="QWEN-ONLY-ID", claude_session="CLAUDE-ONLY-ID")
        lines = content.splitlines()
        in_qwen = False
        for line in lines:
            if line.startswith("## Qwen"):
                in_qwen = True
            elif line.startswith("## ") and in_qwen:
                in_qwen = False
            if in_qwen and "Session id:" in line:
                self.assertNotIn("CLAUDE-ONLY-ID", line,
                                 "Claude session_id leaked into Qwen section")
                self.assertIn("QWEN-ONLY-ID", line,
                              "Qwen section should have Qwen session_id")

    def test_both_sessions_filled_correctly(self) -> None:
        content = self._run_update(qwen_session="QW-123", claude_session="CL-456")
        self.assertIn("- Session id: QW-123", content)
        self.assertIn("- Session id: CL-456", content)
        self.assertEqual(content.count("QW-123"), 1, "Qwen session_id should appear exactly once")
        self.assertEqual(content.count("CL-456"), 1, "Claude session_id should appear exactly once")

    def test_round_counts_filled_per_section(self) -> None:
        content = self._run_update()
        lines = content.splitlines()
        in_qwen = False
        in_claude = False
        for line in lines:
            if line.startswith("## Qwen"):
                in_qwen, in_claude = True, False
            elif line.startswith("## Claude"):
                in_qwen, in_claude = False, True
            elif line.startswith("## "):
                in_qwen, in_claude = False, False
            if "Round count:" in line:
                if in_qwen:
                    self.assertIn("5", line)
                elif in_claude:
                    self.assertIn("8", line)

    def test_passrates_filled(self) -> None:
        content = self._run_update(qwen_passrate="0.5000", claude_passrate="0.9000")
        self.assertIn("- Qwen passrate: 0.5000", content)
        self.assertIn("- Claude passrate: 0.9000", content)

    def test_already_filled_field_not_overwritten(self) -> None:
        """A field that already has a value (non-empty) should not be touched."""
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                metadata_dir = delivery_dir / "metadata"
                metadata_dir.mkdir(parents=True, exist_ok=True)

                prefilled = METADATA_TEMPLATE.format(task_id=task.id).replace(
                    "- Session id:\n- Round count:\n\n## Claude",
                    "- Session id: ORIGINAL-QW\n- Round count:\n\n## Claude",
                )
                md_path = metadata_dir / f"{task.id}.md"
                md_path.write_text(prefilled, encoding="utf-8")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "run", model="qwen", status="done", turns=5)
                state.set(task.id, "run", model="claude", status="done", turns=8)
                state.save()

                rows = [{
                    "id": task.id,
                    "qwen_session_id": "NEW-QW",
                    "claude_session_id": "CL-1",
                    "qwen_passrate": "",
                    "claude_passrate": "",
                }]
                _update_metadata_files(config, [task], rows, state)
                content = md_path.read_text(encoding="utf-8")

        self.assertIn("ORIGINAL-QW", content, "Pre-filled value should be preserved")
        self.assertNotIn("NEW-QW", content, "New value should not overwrite existing one")


if __name__ == "__main__":
    unittest.main()
