"""Tests for ctpipe.health.health_check() and scripts/health_check.sh.

Covers:
  - overall_status verdict: healthy / degraded / critical
  - stage_summary per-stage status fields
  - threshold_violations detection (reuses check_passrate_thresholds)
  - integrity_issues detection (missing files via parse_trajectory / is_complete_rubric)
  - shell script exit codes: 0=healthy, 1=degraded, 2=critical
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

from ctpipe.config import (
    BatchConfig,
    model_stem,
    write_task_manifest,
)
from ctpipe.health import health_check
from ctpipe.state import PipelineState
from conftest import build_config, make_task, write_score, write_trajectory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MODELS = ["qwen", "claude"]
_MODEL_AGNOSTIC = ("prepare", "finalize", "validate")
_MODEL_SPECIFIC = ("run", "collect", "score")


def _write_task_manifest(config: BatchConfig, tasks: list) -> None:
    """Write the task manifest JSON inside delivery_dir/metadata/."""
    config.delivery_dir.mkdir(parents=True, exist_ok=True)
    write_task_manifest(config.task_manifest_path, tasks)


def _set_all_done(state: PipelineState, tid: str, *, qwen_pr: float = 0.6, claude_pr: float = 0.8) -> None:
    """Mark every stage as done for *tid*, with finalize passrates."""
    for stage in _MODEL_AGNOSTIC:
        state.set(tid, stage, status="done")
    for model in _MODELS:
        for stage in _MODEL_SPECIFIC:
            state.set(tid, stage, model=model, status="done")
    state.set(tid, "finalize", qwen_passrate=qwen_pr, claude_passrate=claude_pr)


def _create_healthy_files(config: BatchConfig, tid: str) -> None:
    """Create trajectory JSONL + score TOML for both models."""
    d = config.delivery_dir
    for model in _MODELS:
        write_trajectory(
            d / "trajectories" / model / f"{model_stem(tid, model)}.jsonl",
            f"sess-{tid}-{model}",
            model,
        )
        write_score(d / "scores" / model / f"{model_stem(tid, model)}.quality.toml")


# ---------------------------------------------------------------------------
# health_check() — overall_status verdict
# ---------------------------------------------------------------------------


class HealthCheckOverallStatusTest(unittest.TestCase):
    """Verify the three overall_status values and their trigger conditions."""

    def test_healthy_all_done_no_violations(self) -> None:
        """All stages done, passrates within thresholds, files valid → healthy."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])
                _create_healthy_files(config, "CT-0010")

                state = PipelineState(config.state_path)
                _set_all_done(state, "CT-0010", qwen_pr=0.6, claude_pr=0.8)
                state.save()

                result = health_check(config)

        self.assertEqual(result["overall_status"], "healthy")
        self.assertEqual(result["task_count"], 1)
        self.assertEqual(result["permanently_failed"], [])
        self.assertEqual(result["threshold_violations"], [])
        self.assertEqual(result["integrity_issues"], [])

    def test_degraded_threshold_violation_qwen_too_high(self) -> None:
        """qwen passrate ≥ 0.7 → threshold violation → degraded."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])
                _create_healthy_files(config, "CT-0010")

                state = PipelineState(config.state_path)
                # qwen_pr=0.80 >= THRESHOLD_QWEN_MAX (0.7) → violation
                _set_all_done(state, "CT-0010", qwen_pr=0.80, claude_pr=0.90)
                state.save()

                result = health_check(config)

        self.assertEqual(result["overall_status"], "degraded")
        self.assertTrue(
            any("qwen passrate" in v for v in result["threshold_violations"]),
            f"Expected qwen passrate violation, got: {result['threshold_violations']}",
        )

    def test_degraded_threshold_violation_relative_gain(self) -> None:
        """Relative gain ≤ 25% → threshold violation → degraded."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])
                _create_healthy_files(config, "CT-0010")

                state = PipelineState(config.state_path)
                # (0.74 - 0.60) / 0.60 = 23.3% < 25% → violation
                _set_all_done(state, "CT-0010", qwen_pr=0.60, claude_pr=0.74)
                state.save()

                result = health_check(config)

        self.assertEqual(result["overall_status"], "degraded")
        self.assertTrue(
            any("relative gain" in v for v in result["threshold_violations"]),
            f"Expected relative gain violation, got: {result['threshold_violations']}",
        )

    def test_degraded_integrity_missing_score_file(self) -> None:
        """Score status=done but score file absent → integrity issue → degraded."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])
                # Create trajectories but NOT score files
                for model in _MODELS:
                    write_trajectory(
                        config.delivery_dir / "trajectories" / model / f"{model_stem('CT-0010', model)}.jsonl",
                        f"sess-{model}",
                        model,
                    )

                state = PipelineState(config.state_path)
                for stage in _MODEL_AGNOSTIC:
                    state.set("CT-0010", stage, status="done")
                for model in _MODELS:
                    state.set("CT-0010", "run", model=model, status="done")
                    state.set("CT-0010", "collect", model=model, status="done")
                    # score=done triggers the missing-file check
                    state.set("CT-0010", "score", model=model, status="done")
                state.save()

                result = health_check(config)

        self.assertEqual(result["overall_status"], "degraded")
        self.assertTrue(
            any("score file missing" in i for i in result["integrity_issues"]),
            f"Expected score-file-missing issue, got: {result['integrity_issues']}",
        )
        # No threshold violations because finalize was never set
        self.assertEqual(result["threshold_violations"], [])

    def test_degraded_integrity_trajectory_too_short(self) -> None:
        """Trajectory JSONL with < MIN_TRAJECTORY_LINES → integrity issue → degraded."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])

                # Write a trajectory with only 3 lines (below MIN_TRAJECTORY_LINES)
                for model in _MODELS:
                    traj = config.delivery_dir / "trajectories" / model / f"{model_stem('CT-0010', model)}.jsonl"
                    traj.parent.mkdir(parents=True, exist_ok=True)
                    traj.write_text(
                        '{"type":"system"}\n{"type":"user"}\n{"type":"assistant"}\n',
                        encoding="utf-8",
                    )
                # Write valid score files
                for model in _MODELS:
                    write_score(
                        config.delivery_dir / "scores" / model / f"{model_stem('CT-0010', model)}.quality.toml"
                    )

                state = PipelineState(config.state_path)
                _set_all_done(state, "CT-0010", qwen_pr=0.6, claude_pr=0.8)
                state.save()

                result = health_check(config)

        self.assertEqual(result["overall_status"], "degraded")
        self.assertTrue(
            any("too short" in i for i in result["integrity_issues"]),
            f"Expected trajectory-too-short issue, got: {result['integrity_issues']}",
        )

    def test_critical_permanently_failed(self) -> None:
        """Any permanently_failed entry → critical (overrides everything else)."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])
                _create_healthy_files(config, "CT-0010")

                state = PipelineState(config.state_path)
                _set_all_done(state, "CT-0010", qwen_pr=0.6, claude_pr=0.8)
                # Override one stage to permanently_failed
                state.set("CT-0010", "run", model="qwen",
                          status="permanently_failed")
                state.save()

                result = health_check(config)

        self.assertEqual(result["overall_status"], "critical")
        self.assertEqual(len(result["permanently_failed"]), 1)
        entry = result["permanently_failed"][0]
        self.assertEqual(entry["task_id"], "CT-0010")
        self.assertEqual(entry["stage"], "run")
        self.assertEqual(entry["model"], "qwen")

    def test_critical_permanently_failed_model_agnostic(self) -> None:
        """permanently_failed on a model-agnostic stage (no model key)."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])

                state = PipelineState(config.state_path)
                state.set("CT-0010", "prepare", status="permanently_failed")
                state.save()

                result = health_check(config)

        self.assertEqual(result["overall_status"], "critical")
        self.assertEqual(len(result["permanently_failed"]), 1)
        entry = result["permanently_failed"][0]
        self.assertEqual(entry["stage"], "prepare")
        self.assertNotIn("model", entry)

    def test_critical_delivery_dir_missing(self) -> None:
        """Non-existent delivery directory → critical with integrity_issues message."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                # Do NOT create delivery_dir
                result = health_check(config)

        self.assertEqual(result["overall_status"], "critical")
        self.assertEqual(result["task_count"], 0)
        self.assertTrue(
            any("delivery directory not found" in i for i in result["integrity_issues"]),
        )

    def test_critical_no_tasks(self) -> None:
        """Delivery exists but task manifest is empty → critical."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([], delivery_date="20990101")
                config.delivery_dir.mkdir(parents=True, exist_ok=True)
                # Write an empty manifest
                write_task_manifest(config.task_manifest_path, [])

                result = health_check(config)

        self.assertEqual(result["overall_status"], "critical")
        self.assertEqual(result["task_count"], 0)


# ---------------------------------------------------------------------------
# health_check() — stage_summary
# ---------------------------------------------------------------------------


class HealthCheckStageSummaryTest(unittest.TestCase):
    """Verify stage_summary rows and their per-stage status field."""

    def test_all_healthy_stages(self) -> None:
        """All stages done → every row status = 'healthy'."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])
                _create_healthy_files(config, "CT-0010")

                state = PipelineState(config.state_path)
                _set_all_done(state, "CT-0010", qwen_pr=0.6, claude_pr=0.8)
                state.save()

                result = health_check(config)

        for row in result["stage_summary"]:
            self.assertEqual(
                row["status"], "healthy",
                f"stage {row['stage']} should be healthy, got {row['status']}",
            )
            self.assertEqual(row["done"], row["total"])
            self.assertEqual(row["failed"], 0)

    def test_critical_stage_with_failure(self) -> None:
        """A stage with any failed entry → that row status = 'critical'."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])

                state = PipelineState(config.state_path)
                _set_all_done(state, "CT-0010", qwen_pr=0.6, claude_pr=0.8)
                state.set("CT-0010", "score", model="qwen", status="failed")
                state.save()

                result = health_check(config)

        score_qwen = next(
            r for r in result["stage_summary"] if r["stage"] == "score/qwen"
        )
        self.assertEqual(score_qwen["status"], "critical")
        self.assertEqual(score_qwen["failed"], 1)

    def test_degraded_stage_with_pending(self) -> None:
        """A stage with pending entries (no failures) → that row status = 'degraded'."""
        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])

                state = PipelineState(config.state_path)
                _set_all_done(state, "CT-0010", qwen_pr=0.6, claude_pr=0.8)
                # Reset one stage to pending (no status set)
                state.set("CT-0010", "validate", status="")
                state.save()

                result = health_check(config)

        validate_row = next(
            r for r in result["stage_summary"] if r["stage"] == "validate"
        )
        self.assertEqual(validate_row["status"], "degraded")
        self.assertGreater(validate_row["pending"], 0)


# ---------------------------------------------------------------------------
# health_check() — cross-model integrity
# ---------------------------------------------------------------------------


class HealthCheckCrossModelTest(unittest.TestCase):
    """Verify cross-model scoring consistency checks."""

    def test_criterion_name_mismatch_detected(self) -> None:
        """Different criterion names between qwen and claude → integrity issue."""
        from ctpipe.config import REFERENCE_CRITERION_DESCRIPTIONS, REFERENCE_CRITERION_NAMES
        from ctpipe.toml_utils import Criterion, write_quality_toml

        task = make_task("CT-0010")
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(
                BatchConfig, "base_dir",
                new_callable=PropertyMock, return_value=Path(tmpdir),
            ):
                config = build_config([task], delivery_date="20990101")
                _write_task_manifest(config, [task])

                # Write trajectories
                for model in _MODELS:
                    write_trajectory(
                        config.delivery_dir / "trajectories" / model / f"{model_stem('CT-0010', model)}.jsonl",
                        f"sess-{model}", model,
                    )

                # Write score files with DIFFERENT criterion names
                names = REFERENCE_CRITERION_NAMES[:7]
                for model in _MODELS:
                    score_path = config.delivery_dir / "scores" / model / f"{model_stem('CT-0010', model)}.quality.toml"
                    score_path.parent.mkdir(parents=True, exist_ok=True)
                    criteria = [
                        Criterion(
                            name + ("_qwen" if model == "qwen" else "_claude"),
                            REFERENCE_CRITERION_DESCRIPTIONS[name],
                            "likert", 5, 1.0, 3, "ok",
                        )
                        for name in names
                    ]
                    write_quality_toml(score_path, criteria)

                state = PipelineState(config.state_path)
                _set_all_done(state, "CT-0010", qwen_pr=0.6, claude_pr=0.8)
                state.save()

                result = health_check(config)

        self.assertTrue(
            any("criterion name mismatch" in i for i in result["integrity_issues"]),
            f"Expected criterion mismatch, got: {result['integrity_issues']}",
        )


# ---------------------------------------------------------------------------
# Shell script exit codes (subprocess integration)
# ---------------------------------------------------------------------------


class HealthCheckShellScriptTest(unittest.TestCase):
    """Test scripts/health_check.sh exit code mapping via subprocess.

    Creates a minimal project layout (tasks.toml + .env + delivery dir with
    pipeline state) and invokes the shell script end-to-end.
    """

    _PROJECT_ROOT = Path(__file__).resolve().parent.parent

    _TASKS_TOML = """\
[batch]
delivery_date = "20990101"
runs_root = "D:/runs"

[[task]]
id = "CT-0099"
project_path = "D:/projects/demo"
task_type = "bug-fix"
domain = "web_frontend"
language = "ts"
prompt_qwen = "fix it"
prompt_claude = "fix it"
"""

    @classmethod
    def setUpClass(cls) -> None:
        import shutil
        # Prefer Git Bash (supports /d/ paths); fall back to any bash.
        for candidate in ("bash", ):
            path = shutil.which(candidate)
            if path:
                # Verify it's not WSL bash (which needs /mnt/d/ not /d/)
                proc = subprocess.run(
                    [path, "-c", "echo test"],
                    capture_output=True, text=True, timeout=5,
                )
                if proc.returncode == 0 and "test" in proc.stdout:
                    cls._BASH = path
                    return
        raise unittest.SkipTest("bash not available")

    @staticmethod
    def _to_bash_path(p: Path) -> str:
        """Convert a Windows path to Git Bash format (D:\\x → /d/x).

        If running under WSL bash, converts to /mnt/d/x instead.
        """
        import re
        s = str(p).replace("\\", "/")
        s = re.sub(r"^([A-Za-z]):", lambda m: "/" + m.group(1).lower(), s)
        return s

    def _make_tmpdir(self):
        """Context manager for a temp dir that tolerates Windows cleanup errors."""
        return tempfile.TemporaryDirectory(ignore_cleanup_errors=True)

    def _write_layout(self, tmpdir: Path, state_setup) -> tuple[Path, Path]:
        """Write tasks.toml, .env, delivery dir, manifest, and state."""
        tasks_toml = tmpdir / "tasks.toml"
        tasks_toml.write_text(self._TASKS_TOML, encoding="utf-8")
        (tmpdir / ".env").write_text("", encoding="utf-8")

        from ctpipe.config import load_config
        config = load_config(tasks_toml, tmpdir / ".env")

        config.delivery_dir.mkdir(parents=True, exist_ok=True)
        task = make_task("CT-0099")
        write_task_manifest(config.task_manifest_path, [task])
        _create_healthy_files(config, "CT-0099")

        state = PipelineState(config.state_path)
        state_setup(config, state)
        state.save()

        return tasks_toml, tmpdir / ".env"

    def _run_script(
        self, tasks_toml: Path, env_path: Path, delivery_date: str = "20990101",
    ) -> subprocess.CompletedProcess:
        bp = self._to_bash_path
        return subprocess.run(
            [
                self._BASH, bp(self._PROJECT_ROOT / "scripts" / "health_check.sh"),
                "--config", bp(tasks_toml),
                "--env", bp(env_path),
                "--delivery-date", delivery_date,
                "--output", bp(tasks_toml.parent / "hc_report.json"),
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=str(self._PROJECT_ROOT),
            timeout=30,
        )

    def test_healthy_exit_0(self) -> None:
        """All done, no violations → exit code 0."""
        def setup(config, state):
            _set_all_done(state, "CT-0099", qwen_pr=0.6, claude_pr=0.8)

        with self._make_tmpdir() as tmpdir:
            tasks_toml, env_path = self._write_layout(Path(tmpdir), setup)
            proc = self._run_script(tasks_toml, env_path)

        self.assertEqual(proc.returncode, 0, f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
        self.assertIn("HEALTHY", proc.stdout)

    def test_degraded_exit_1(self) -> None:
        """Threshold violation → exit code 1."""
        def setup(config, state):
            _set_all_done(state, "CT-0099", qwen_pr=0.80, claude_pr=0.90)

        with self._make_tmpdir() as tmpdir:
            tasks_toml, env_path = self._write_layout(Path(tmpdir), setup)
            proc = self._run_script(tasks_toml, env_path)

        self.assertEqual(proc.returncode, 1, f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
        self.assertIn("DEGRADED", proc.stdout)

    def test_critical_exit_2(self) -> None:
        """Permanently failed → exit code 2."""
        def setup(config, state):
            _set_all_done(state, "CT-0099", qwen_pr=0.6, claude_pr=0.8)
            state.set("CT-0099", "run", model="qwen", status="permanently_failed")

        with self._make_tmpdir() as tmpdir:
            tasks_toml, env_path = self._write_layout(Path(tmpdir), setup)
            proc = self._run_script(tasks_toml, env_path)

        self.assertEqual(proc.returncode, 2, f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
        self.assertIn("CRITICAL", proc.stdout)

    def test_critical_missing_delivery_exit_2(self) -> None:
        """Non-existent delivery date → exit code 2."""
        def setup(config, state):
            pass

        with self._make_tmpdir() as tmpdir:
            tasks_toml, env_path = self._write_layout(Path(tmpdir), setup)
            proc = self._run_script(tasks_toml, env_path, delivery_date="99991231")

        self.assertEqual(proc.returncode, 2, f"stdout: {proc.stdout}\nstderr: {proc.stderr}")
        self.assertIn("CRITICAL", proc.stdout)

    def test_json_report_written(self) -> None:
        """--output flag writes a valid JSON file with overall_status."""
        def setup(config, state):
            _set_all_done(state, "CT-0099", qwen_pr=0.6, claude_pr=0.8)

        with self._make_tmpdir() as tmpdir:
            tasks_toml, env_path = self._write_layout(Path(tmpdir), setup)
            out_path = Path(tmpdir) / "hc_report.json"
            proc = self._run_script(tasks_toml, env_path)

            self.assertEqual(proc.returncode, 0, f"stderr: {proc.stderr}")
            self.assertTrue(out_path.exists(), "JSON report file was not created")
            data = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertIn("overall_status", data)
            self.assertEqual(data["overall_status"], "healthy")


if __name__ == "__main__":
    unittest.main()
