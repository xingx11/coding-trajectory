"""Unit tests for ctpipe.export — structured JSON report generation.

Covers three scenarios:
1. Normal export with complete data for all tasks and models
2. Partial data: missing score files, missing trajectories → null fields
3. Empty task list → empty report with zero counts
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from ctpipe.config import (
    REFERENCE_CRITERION_DESCRIPTIONS,
    REFERENCE_CRITERION_NAMES,
    BatchConfig,
    ModelConfig,
    TaskConfig,
    write_task_manifest,
)
from ctpipe.export import export_report
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
        person_id="42",
    )


def _make_task(task_id: str = "CT-0001", task_type: str = "feature") -> TaskConfig:
    return TaskConfig(
        id=task_id,
        project_path=Path("D:/projects/demo"),
        clone_method="git",
        task_type=task_type,
        domain="web_backend",
        language="python",
        prompt_qwen="qwen prompt",
        prompt_claude="claude prompt",
        bad_pattern="lazy_shortcut",
    )


def _setup_full_data(
    delivery_dir: Path,
    state: PipelineState,
    task_id: str,
    model_name: str,
    score: int = 4,
    session_id: str = "sess-001",
    turns: int = 3,
    duration_s: float = 120.5,
) -> None:
    """Create trajectory, score file, and state entries for one task/model."""
    # Trajectory
    traj_dir = delivery_dir / "trajectories" / model_name
    traj_dir.mkdir(parents=True, exist_ok=True)
    traj_file = traj_dir / f"{task_id}.jsonl"
    lines = [json.dumps({"sessionId": session_id, "type": "user"}) + "\n"]
    lines.append(json.dumps({"type": "assistant", "message": {"role": "assistant", "model": f"{model_name}-test"}}) + "\n")
    lines.append(json.dumps({"type": "user"}) + "\n")
    traj_file.write_text("".join(lines), encoding="utf-8")

    # Score file
    score_dir = delivery_dir / "scores" / model_name
    score_dir.mkdir(parents=True, exist_ok=True)
    names = REFERENCE_CRITERION_NAMES[:7]
    criteria = [
        Criterion(
            name, REFERENCE_CRITERION_DESCRIPTIONS[name], "likert", 5,
            1.0, score, "Good work"
        )
        for name in names
    ]
    write_quality_toml(score_dir / f"{task_id}.quality.toml", criteria)

    # State
    state.set(
        task_id, "run", model=model_name,
        status="done", session_id=session_id, turns=turns, duration_s=duration_s,
    )
    state.set(
        task_id, "collect", model=model_name,
        status="done", session_id=session_id,
        jsonl_path=f"trajectories/{model_name}/{task_id}.jsonl",
    )


# =========================================================================
# Test 1: Normal export with complete data
# =========================================================================


class ExportNormalTest(unittest.TestCase):

    def test_full_report_structure(self) -> None:
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
                    _setup_full_data(delivery_dir, state, task.id, "qwen", score=2, session_id=f"qw-{task.id}")
                    _setup_full_data(delivery_dir, state, task.id, "claude", score=4, session_id=f"cl-{task.id}")
                state.save()

                report = export_report(config)

        # Top-level keys
        self.assertEqual(set(report.keys()), {"batch_info", "tasks", "summary"})

        # batch_info
        bi = report["batch_info"]
        self.assertEqual(bi["delivery_date"], "20990101")
        self.assertEqual(bi["person_id"], "42")
        self.assertEqual(bi["task_count"], 3)

        # tasks array
        self.assertEqual(len(report["tasks"]), 3)
        for entry in report["tasks"]:
            self.assertEqual(set(entry.keys()), {"metadata", "trajectory_info", "scoring", "threshold_check"})

            # metadata
            md = entry["metadata"]
            self.assertIn(md["id"], [t.id for t in tasks])
            self.assertEqual(md["task_type"], "feature")
            self.assertEqual(md["domain"], "web_backend")
            self.assertEqual(md["language"], "python")
            self.assertEqual(md["bad_pattern"], "lazy_shortcut")

            # trajectory_info: both models present
            ti = entry["trajectory_info"]
            self.assertIn("qwen", ti)
            self.assertIn("claude", ti)
            for model_name in ("qwen", "claude"):
                model_ti = ti[model_name]
                self.assertIsNotNone(model_ti)
                self.assertIn("session_id", model_ti)
                self.assertIn("turns", model_ti)
                self.assertIn("duration_s", model_ti)
                self.assertIsNotNone(model_ti["session_id"])
                self.assertEqual(model_ti["turns"], 3)
                self.assertAlmostEqual(model_ti["duration_s"], 120.5, places=1)

            # scoring: both models present
            sc = entry["scoring"]
            for model_name in ("qwen", "claude"):
                model_sc = sc[model_name]
                self.assertIsNotNone(model_sc)
                self.assertIsInstance(model_sc["criteria"], list)
                self.assertEqual(len(model_sc["criteria"]), 7)
                self.assertIsNotNone(model_sc["passrate"])
                for crit in model_sc["criteria"]:
                    self.assertIn("name", crit)
                    self.assertIn("score", crit)
                    self.assertIn("weight", crit)
                    self.assertIn("rationale", crit)

            # qwen score=2 → passrate=0.4, claude score=4 → passrate=0.8
            self.assertAlmostEqual(sc["qwen"]["passrate"], 0.4, places=3)
            self.assertAlmostEqual(sc["claude"]["passrate"], 0.8, places=3)

            # threshold_check
            tc = entry["threshold_check"]
            self.assertIn("passed", tc)
            self.assertIn("issues", tc)

        # summary
        summary = report["summary"]
        self.assertEqual(summary["total_tasks"], 3)
        self.assertIn("threshold_passed", summary)
        self.assertIn("per_model_passrate", summary)
        # Both models should have stats
        for model_name in ("qwen", "claude"):
            pm = summary["per_model_passrate"][model_name]
            self.assertIsNotNone(pm)
            self.assertIn("min", pm)
            self.assertIn("max", pm)
            self.assertIn("mean", pm)
            self.assertEqual(pm["count"], 3)

    def test_threshold_check_passed_when_claude_beats_qwen(self) -> None:
        """qwen passrate=0.4, claude passrate=0.8, gain=100% → all thresholds met."""
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                _setup_full_data(delivery_dir, state, task.id, "qwen", score=2)
                _setup_full_data(delivery_dir, state, task.id, "claude", score=4)
                state.save()

                report = export_report(config)

        tc = report["tasks"][0]["threshold_check"]
        self.assertTrue(tc["passed"])
        self.assertIsNone(tc["issues"])

    def test_threshold_check_failed_when_qwen_too_high(self) -> None:
        """qwen score=4 → 0.8 >= 0.7 threshold → issues."""
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                _setup_full_data(delivery_dir, state, task.id, "qwen", score=4)
                _setup_full_data(delivery_dir, state, task.id, "claude", score=5)
                state.save()

                report = export_report(config)

        tc = report["tasks"][0]["threshold_check"]
        self.assertFalse(tc["passed"])
        self.assertIsInstance(tc["issues"], list)
        self.assertTrue(len(tc["issues"]) > 0)


# =========================================================================
# Test 2: Partial data — missing files → null fields, no errors
# =========================================================================


class ExportPartialMissingTest(unittest.TestCase):

    def test_missing_score_file_sets_scoring_null(self) -> None:
        """When a score file does not exist, scoring for that model is null."""
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                # Only set up qwen data, skip claude score file
                _setup_full_data(delivery_dir, state, task.id, "qwen", score=3)
                # Claude: only trajectory, no score file
                traj_dir = delivery_dir / "trajectories" / "claude"
                traj_dir.mkdir(parents=True, exist_ok=True)
                (traj_dir / f"{task.id}.jsonl").write_text(
                    json.dumps({"sessionId": "cl-sess"}) + "\n", encoding="utf-8",
                )
                state.set(task.id, "run", model="claude", status="done", session_id="cl-sess", turns=2)
                state.save()

                report = export_report(config)

        entry = report["tasks"][0]
        self.assertIsNotNone(entry["scoring"]["qwen"])
        self.assertIsNone(entry["scoring"]["claude"])

    def test_missing_trajectory_sets_trajectory_info_null(self) -> None:
        """When no trajectory file and no state data exists, trajectory_info is null."""
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                # No data at all for either model
                state.save()

                report = export_report(config)

        entry = report["tasks"][0]
        self.assertIsNone(entry["trajectory_info"]["qwen"])
        self.assertIsNone(entry["trajectory_info"]["claude"])
        self.assertIsNone(entry["scoring"]["qwen"])
        self.assertIsNone(entry["scoring"]["claude"])

    def test_partial_trajectory_with_missing_fields(self) -> None:
        """State has session_id but no turns/duration → only session_id filled."""
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                # Only session_id, no turns or duration
                state.set(task.id, "run", model="qwen", status="done", session_id="qw-partial")
                state.set(task.id, "collect", model="qwen", status="done", session_id="qw-partial")
                state.save()

                report = export_report(config)

        ti = report["tasks"][0]["trajectory_info"]["qwen"]
        self.assertIsNotNone(ti)
        self.assertEqual(ti["session_id"], "qw-partial")
        self.assertIsNone(ti["turns"])
        self.assertIsNone(ti["duration_s"])

    def test_report_does_not_raise_on_completely_empty_delivery(self) -> None:
        """Empty delivery dir (no trajectories, no scores, no state) should not raise."""
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                # No pipeline_state.json, no files at all
                report = export_report(config)

        self.assertEqual(len(report["tasks"]), 1)
        entry = report["tasks"][0]
        self.assertIsNone(entry["trajectory_info"]["qwen"])
        self.assertIsNone(entry["trajectory_info"]["claude"])
        self.assertIsNone(entry["scoring"]["qwen"])
        self.assertIsNone(entry["scoring"]["claude"])
        self.assertFalse(entry["threshold_check"]["passed"])


# =========================================================================
# Test 3: Empty task list → empty report
# =========================================================================


class ExportEmptyTaskListTest(unittest.TestCase):

    def test_empty_tasks_produces_empty_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                report = export_report(config)

        self.assertEqual(report["batch_info"]["task_count"], 0)
        self.assertEqual(report["tasks"], [])
        self.assertEqual(report["summary"]["total_tasks"], 0)
        self.assertEqual(report["summary"]["threshold_passed"], 0)

    def test_empty_tasks_summary_has_null_model_passrates(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                report = export_report(config)

        for model_name in ("qwen", "claude"):
            self.assertIsNone(report["summary"]["per_model_passrate"][model_name])


# =========================================================================
# Test 4: Output to file
# =========================================================================


class ExportOutputFileTest(unittest.TestCase):

    def test_output_writes_json_to_file(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                _setup_full_data(delivery_dir, state, task.id, "qwen", score=3)
                _setup_full_data(delivery_dir, state, task.id, "claude", score=4)
                state.save()

                output_path = temp_base / "reports" / "test_report.json"
                report = export_report(config, output=output_path)

                self.assertTrue(output_path.exists())
                loaded = json.loads(output_path.read_text(encoding="utf-8"))
                self.assertEqual(loaded["batch_info"], report["batch_info"])
                self.assertEqual(len(loaded["tasks"]), 1)

    def test_output_creates_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                output_path = temp_base / "a" / "b" / "c" / "report.json"
                export_report(config, output=output_path)

                self.assertTrue(output_path.exists())


# =========================================================================
# Test 5: Selective task/model filtering
# =========================================================================


class ExportFilteringTest(unittest.TestCase):

    def test_task_ids_filter(self) -> None:
        tasks = [_make_task(f"CT-{i:04d}") for i in range(1, 4)]
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config(tasks)
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, tasks)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.save()

                report = export_report(config, task_ids=["CT-0002"])

        self.assertEqual(report["batch_info"]["task_count"], 1)
        self.assertEqual(len(report["tasks"]), 1)
        self.assertEqual(report["tasks"][0]["metadata"]["id"], "CT-0002")

    def test_single_model_filter(self) -> None:
        task = _make_task()
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config([task])
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, [task])

                state = PipelineState(delivery_dir / "pipeline_state.json")
                _setup_full_data(delivery_dir, state, task.id, "qwen", score=3)
                _setup_full_data(delivery_dir, state, task.id, "claude", score=4)
                state.save()

                report = export_report(config, models=["qwen"])

        entry = report["tasks"][0]
        self.assertIn("qwen", entry["trajectory_info"])
        self.assertNotIn("claude", entry["trajectory_info"])
        self.assertIn("qwen", entry["scoring"])
        self.assertNotIn("claude", entry["scoring"])


# =========================================================================
# Test 6: Summary statistics correctness
# =========================================================================


class ExportSummaryStatsTest(unittest.TestCase):

    def test_summary_passrate_min_max_mean(self) -> None:
        """Three tasks with different scores → correct min/max/mean."""
        tasks = [_make_task(f"CT-{i:04d}") for i in range(1, 4)]
        # score=2→0.4, score=3→0.6, score=4→0.8
        scores = [2, 3, 4]

        with tempfile.TemporaryDirectory() as tmpdir:
            temp_base = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=temp_base):
                config = _build_config(tasks)
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)
                write_task_manifest(config.task_manifest_path, tasks)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                for task, score in zip(tasks, scores):
                    _setup_full_data(delivery_dir, state, task.id, "qwen", score=score)
                    _setup_full_data(delivery_dir, state, task.id, "claude", score=score)
                state.save()

                report = export_report(config)

        for model_name in ("qwen", "claude"):
            pm = report["summary"]["per_model_passrate"][model_name]
            self.assertAlmostEqual(pm["min"], 0.4, places=3)
            self.assertAlmostEqual(pm["max"], 0.8, places=3)
            self.assertAlmostEqual(pm["mean"], 0.6, places=3)
            self.assertEqual(pm["count"], 3)


if __name__ == "__main__":
    unittest.main()
