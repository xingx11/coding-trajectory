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
    SUBMISSION_KEY_MAP,
    REFERENCE_CRITERION_NAMES,
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
        bad_pattern="lazy_shortcut",
    )


def _setup_task_scores(
    delivery_dir: Path,
    state: PipelineState,
    task_id: str,
    model_name: str,
    score_per_criterion: int,
    session_id: str = "s1",
) -> None:
    """Create trajectory file, score file, and state entries for one task/model.

    score_per_criterion: integer 1-5. When all criteria have the same score,
    passrate = score_per_criterion / 5  (e.g. score=2 → passrate=0.4).
    """
    traj_dir = delivery_dir / "trajectories" / model_name
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_file = traj_dir / f"{task_id}.jsonl"
    traj_file.write_text(
        json.dumps({"sessionId": session_id}) + "\n",
        encoding="utf-8",
    )

    score_dir = delivery_dir / "scores" / model_name
    score_dir.mkdir(parents=True, exist_ok=True)
    names = REFERENCE_CRITERION_NAMES[:7]
    criteria = [
        Criterion(
            name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
            1.0, score_per_criterion, "评分理由"
        )
        for name in names
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
                    _setup_task_scores(delivery_dir, state, task.id, "qwen", 2, f"qw-{task.id}")
                    _setup_task_scores(delivery_dir, state, task.id, "claude", 4, f"cl-{task.id}")
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
        # finalize() maps CT-XXXX → formatted submission IDs
        # person_id=99, date=20990101 → "99-101-bug-fix-{01,02,03}"
        self.assertEqual(task_ids, {"99-101-bug-fix-01", "99-101-bug-fix-02", "99-101-bug-fix-03"})


# =========================================================================
# Dual-model thresholds
# =========================================================================


class FinalizeDualModelThresholdTest(unittest.TestCase):
    """Test threshold checks with score_per_criterion (int 1-5).

    passrate = score_per_criterion / 5:
      score=1 → 0.2, score=2 → 0.4, score=3 → 0.6, score=4 → 0.8, score=5 → 1.0
    """

    def _run_finalize_and_get_status(
        self,
        qwen_score: int,
        claude_score: int,
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
                _setup_task_scores(delivery_dir, state, task.id, "qwen", qwen_score, "qw-s1")
                _setup_task_scores(delivery_dir, state, task.id, "claude", claude_score, "cl-s1")
                state.save()

                finalize(config)

                state2 = PipelineState(delivery_dir / "pipeline_state.json")
                return state2.get(task.id, "finalize").get("status", "")

    def test_qwen_passrate_above_threshold_is_partial(self) -> None:
        # qwen score=4 → passrate=0.8 >= 0.7 → partial
        status = self._run_finalize_and_get_status(qwen_score=4, claude_score=5)
        self.assertEqual(status, "partial")

    def test_qwen_below_threshold_with_sufficient_gain_is_done(self) -> None:
        # qwen score=3 → 0.6 < 0.7, claude score=4 → 0.8
        # gain = (0.8-0.6)/0.6 = 33.3% > 30% → done
        status = self._run_finalize_and_get_status(qwen_score=3, claude_score=4)
        self.assertEqual(status, "done")

    def test_relative_gain_below_threshold_is_partial(self) -> None:
        # qwen score=3 → 0.6, claude score=3 → 0.6, gain=0% <= 25% → partial
        status = self._run_finalize_and_get_status(qwen_score=3, claude_score=3)
        self.assertEqual(status, "partial")

    def test_claude_low_but_gain_sufficient_is_partial(self) -> None:
        # qwen score=1 → 0.2, claude score=3 → 0.6
        # gain = 200% > 25%, but claude 0.6 <= 0.7 threshold → partial
        status = self._run_finalize_and_get_status(qwen_score=1, claude_score=3)
        self.assertEqual(status, "partial")

    def test_claude_leq_qwen_is_partial(self) -> None:
        # claude score=2 → 0.4 <= qwen score=2 → 0.4 → partial
        status = self._run_finalize_and_get_status(qwen_score=2, claude_score=2)
        self.assertEqual(status, "partial")

    def test_all_thresholds_met_is_done(self) -> None:
        # qwen score=2 → 0.4 < 0.7, claude score=4 → 0.8 > 0.7
        # gain = (0.8-0.4)/0.4 = 100% > 25% → done
        status = self._run_finalize_and_get_status(qwen_score=2, claude_score=4)
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


# =========================================================================
# Regression: finalize --tasks must keep full-list submission ID numbering
# =========================================================================


class FinalizePartialTasksIdConsistencyTest(unittest.TestCase):
    """Regression: finalize(config, task_ids=['CT-0002']) must assign the
    same submission ID as a full finalize over all tasks."""

    def test_second_task_keeps_full_run_id(self) -> None:
        """3 same-type tasks: partial finalize of CT-0002 must get seq=02."""
        tasks = [_make_task(f"CT-{i:04d}") for i in range(1, 4)]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(
                BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base
            ):
                config = _build_config(tasks)
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, tasks)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                for task in tasks:
                    _setup_task_scores(
                        delivery_dir, state, task.id, "qwen", 2, f"qw-{task.id}"
                    )
                    _setup_task_scores(
                        delivery_dir, state, task.id, "claude", 4, f"cl-{task.id}"
                    )
                state.save()

                # Full finalize → read CT-0002's ID from CSV
                buf = io.StringIO()
                with redirect_stdout(buf):
                    finalize(config)

                with (delivery_dir / "submission.csv").open(
                    "r", encoding="utf-8-sig", newline=""
                ) as f:
                    full_rows = {
                        r["_task_id"] if "_task_id" in r else r["id"]: r
                        for r in csv.DictReader(f)
                    }

                # Read the internal-key CSV to find CT-0002's full-run ID.
                # _write_submission_csv maps internal keys → Chinese column names.
                # The task_id is not directly in the CSV, so parse from output.
                full_output = buf.getvalue()
                # Output lines look like: "  99-101-bug-fix-02 (CT-0002): ..."
                full_id_for_ct0002 = None
                for line in full_output.splitlines():
                    if "(CT-0002)" in line:
                        full_id_for_ct0002 = line.strip().split()[0]
                        break
                self.assertIsNotNone(full_id_for_ct0002, "CT-0002 not in full output")
                self.assertIn("bug-fix-02", full_id_for_ct0002)

                # Partial finalize of CT-0002 only
                buf2 = io.StringIO()
                with redirect_stdout(buf2):
                    finalize(config, task_ids=["CT-0002"])

                partial_output = buf2.getvalue()
                partial_id_for_ct0002 = None
                for line in partial_output.splitlines():
                    if "(CT-0002)" in line:
                        partial_id_for_ct0002 = line.strip().split()[0]
                        break

                self.assertEqual(
                    partial_id_for_ct0002,
                    full_id_for_ct0002,
                    "Partial finalize changed submission ID for CT-0002",
                )

                # Also verify the overwritten CSV has the correct ID
                with (delivery_dir / "submission.csv").open(
                    "r", encoding="utf-8-sig", newline=""
                ) as f:
                    partial_rows = list(csv.DictReader(f))
                self.assertEqual(len(partial_rows), 1)
                self.assertEqual(partial_rows[0]["id"], full_id_for_ct0002)


if __name__ == "__main__":
    unittest.main()
