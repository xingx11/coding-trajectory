from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, PropertyMock

from conftest import build_config, make_task

from ctpipe.collect import collect_single
from ctpipe.config import BatchConfig, ModelConfig, TaskConfig
from ctpipe.state import PipelineState
from ctpipe.trajectory import TrajectoryInfo



def _write_jsonl(path: Path, session_id: str, model: str, lines: int = 15) -> None:
    """Write a realistic JSONL trajectory file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"sessionId": session_id, "type": "system", "timestamp": "2025-01-01T00:00:00Z"}) + "\n")
        f.write(json.dumps({"type": "user", "timestamp": "2025-01-01T00:00:01Z", "message": {"role": "user", "content": "hello"}}) + "\n")
        for i in range(lines - 2):
            f.write(json.dumps({
                "type": "assistant",
                "timestamp": f"2025-01-01T00:00:{i+2:02d}Z",
                "message": {"role": "assistant", "model": model, "content": f"response {i}"},
            }) + "\n")


class CollectNormalPathTest(unittest.TestCase):
    """Normal collect: run status=done, start_time and session_id present."""

    def test_normal_collect_succeeds_with_status_done(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                jsonl_path = tmp / "trajectory.jsonl"
                _write_jsonl(jsonl_path, session_id="sess-123", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude", status="done", session_id="sess-123", start_time=1000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=jsonl_path):
                    result = collect_single(task, "claude", config, state)

        self.assertTrue(result)
        collect_info = state.get(task.id, "collect", "claude")
        self.assertEqual(collect_info["status"], "done")
        self.assertFalse(collect_info.get("recovery", False))
        self.assertFalse(collect_info.get("salvaged", False))

    def test_normal_collect_skips_when_run_not_done(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(tmp / "run_claude"))
                state.set(task.id, "run", model="claude", status="pending")
                state.save()

                result = collect_single(task, "claude", config, state, salvage=False)

        self.assertFalse(result)


class CollectMissingStartTimeTest(unittest.TestCase):
    """Recovery when start_time is missing: infer from .claude/ file mtimes."""

    def test_infers_start_time_from_claude_dir_mtime(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()
                claude_dir = run_dir / ".claude"
                claude_dir.mkdir()
                marker = claude_dir / "settings.json"
                marker.write_text("{}", encoding="utf-8")

                jsonl_path = tmp / "trajectory.jsonl"
                _write_jsonl(jsonl_path, session_id="sess-abc", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude", status="done", session_id="sess-abc")
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=jsonl_path) as mock_find:
                    result = collect_single(task, "claude", config, state)

                called_start_time = mock_find.call_args[0][1]
                self.assertGreater(called_start_time, 0.0, "Should have inferred start_time from .claude/ mtime, not epoch")

        self.assertTrue(result)
        collect_info = state.get(task.id, "collect", "claude")
        self.assertEqual(collect_info["status"], "done")

    def test_falls_back_to_epoch_when_no_claude_dir(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                jsonl_path = tmp / "trajectory.jsonl"
                _write_jsonl(jsonl_path, session_id="sess-xyz", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude", status="done", session_id="sess-xyz")
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=jsonl_path) as mock_find:
                    result = collect_single(task, "claude", config, state)

                called_start_time = mock_find.call_args[0][1]
                self.assertEqual(called_start_time, 0.0, "Should fall back to epoch when .claude/ absent")

        self.assertTrue(result)


class CollectMissingSessionIdTest(unittest.TestCase):
    """Recovery when session_id is missing: infer from newest JSONL in project hash dir."""

    def test_infers_session_id_from_project_hash_dir(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                proj_hash_dir = tmp / "proj_hash"
                proj_hash_dir.mkdir()
                hash_jsonl = proj_hash_dir / "inferred-sess.jsonl"
                _write_jsonl(hash_jsonl, session_id="inferred-sess", model="claude-3.5-sonnet-20241022")

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="inferred-sess", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude", status="done", start_time=0.0)
                state.save()

                with patch("ctpipe.collect.project_hash_dir", return_value=proj_hash_dir):
                    with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl) as mock_find:
                        result = collect_single(task, "claude", config, state)

                    called_session_id = mock_find.call_args[0][2]
                    self.assertEqual(called_session_id, "inferred-sess",
                                     "Should pass inferred session_id to find_trajectory_for_run")

        self.assertTrue(result)
        collect_info = state.get(task.id, "collect", "claude")
        self.assertEqual(collect_info["status"], "done")
        self.assertEqual(collect_info["session_id"], "inferred-sess")

    def test_proceeds_without_session_id_when_inference_fails(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                proj_hash_dir = tmp / "proj_hash_empty"
                proj_hash_dir.mkdir()

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="real-sess", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude", status="done", start_time=1000.0)
                state.save()

                with patch("ctpipe.collect.project_hash_dir", return_value=proj_hash_dir):
                    with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl) as mock_find:
                        result = collect_single(task, "claude", config, state)

                    called_session_id = mock_find.call_args[0][2]
                    self.assertIsNone(called_session_id,
                                     "Should pass None when session_id inference fails")

        self.assertTrue(result)


class CollectSalvageFromInterruptedRunTest(unittest.TestCase):
    """Salvage mode: run status is 'running' (interrupted), collect recovers partial data."""

    def test_salvage_from_running_status_marks_partial(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()
                claude_dir = run_dir / ".claude"
                claude_dir.mkdir()
                (claude_dir / "config.json").write_text("{}", encoding="utf-8")

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="salvage-sess", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude", status="running", session_id="salvage-sess")
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state, salvage=True)

        self.assertTrue(result)
        collect_info = state.get(task.id, "collect", "claude")
        self.assertEqual(collect_info["status"], "partial")
        self.assertTrue(collect_info["recovery"])
        self.assertTrue(collect_info["salvaged"])
        self.assertEqual(collect_info["run_status_at_collect"], "running")

    def test_salvage_disabled_skips_interrupted_run(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(tmp / "run_claude"))
                state.set(task.id, "run", model="claude", status="running")
                state.save()

                result = collect_single(task, "claude", config, state, salvage=False)

        self.assertFalse(result)

    def test_salvage_from_failed_status_marks_partial_with_recovery(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="fail-sess", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude",
                          status="failed", session_id="fail-sess", start_time=1000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state, salvage=True)

        self.assertTrue(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "partial")
        self.assertTrue(info["recovery"])
        self.assertTrue(info["salvaged"])
        self.assertFalse(info["forced"])
        self.assertEqual(info["run_status_at_collect"], "failed")

    def test_salvage_from_timeout_status_marks_partial(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="to-sess", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude",
                          status="timeout", session_id="to-sess", start_time=2000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state)

        self.assertTrue(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "partial")
        self.assertTrue(info["recovery"])

    def test_salvage_short_trajectory_passes_lowered_threshold(self) -> None:
        """Salvage lowers min lines (3 vs 10): a 4-line trajectory should pass."""
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "short.jsonl"
                traj_jsonl.parent.mkdir(parents=True, exist_ok=True)
                with traj_jsonl.open("w", encoding="utf-8") as f:
                    f.write(json.dumps({"sessionId": "short-s", "type": "user"}) + "\n")
                    f.write(json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-x", "content": "a"}}) + "\n")
                    f.write(json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-x", "content": "b"}}) + "\n")
                    f.write(json.dumps({"type": "assistant", "message": {"role": "assistant", "model": "claude-x", "content": "c"}}) + "\n")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude",
                          status="failed", session_id="short-s", start_time=3000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state, salvage=True)

        self.assertTrue(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "partial")
        self.assertTrue(info["recovery"])
        self.assertEqual(info["line_count"], 4)

    def test_salvage_session_id_mismatch_is_warning_not_failure(self) -> None:
        """In salvage mode, session_id mismatch logs a warning but continues."""
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="actual-sid", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                # run expects "expected-sid" but trajectory has "actual-sid"
                state.set(task.id, "run", model="claude",
                          status="failed", session_id="expected-sid", start_time=4000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state, salvage=True)

        self.assertTrue(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "partial")
        self.assertTrue(info["recovery"])
        self.assertEqual(info["session_id"], "actual-sid")


class CollectForceRecoveryTest(unittest.TestCase):
    """Force recovery via --force: bypass start_time/session_id validation."""

    def test_force_with_interrupted_run_marks_partial_and_recovery(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="forced-sess", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                # No start_time, no session_id — would fail normal validation
                state.set(task.id, "run", model="claude", status="failed")
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state, force=True)

        self.assertTrue(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "partial")
        self.assertTrue(info["recovery"])
        self.assertTrue(info["forced"])
        self.assertTrue(info["salvaged"])

    def test_force_with_done_run_recollects_as_done(self) -> None:
        """--force on a done run re-collects as done (not salvage), but recovery=True (force flag)."""
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="new-sess", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude",
                          status="done", session_id="old-sess", start_time=5000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state, force=True)

        self.assertTrue(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "done")
        # recovery = is_salvage or force = False or True = True (force flag sets it)
        self.assertTrue(info["recovery"])
        self.assertTrue(info["forced"])
        self.assertFalse(info["salvaged"])

    def test_force_bypasses_start_time_validation(self) -> None:
        """--force sets start_time=0 and session_id=None for find_trajectory_for_run."""
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="auto", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude", status="error")
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl) as mock_find:
                    result = collect_single(task, "claude", config, state, force=True)

                # start_time should be 0.0, session_id should be None
                call_args = mock_find.call_args[0]
                self.assertEqual(call_args[1], 0.0)
                self.assertIsNone(call_args[2])

        self.assertTrue(result)
        info = state.get(task.id, "collect", "claude")
        self.assertTrue(info["recovery"])
        self.assertTrue(info["forced"])

    def test_force_ignores_already_done_check(self) -> None:
        """--force must re-collect even when collect state is already done."""
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "trajectory.jsonl"
                _write_jsonl(traj_jsonl, session_id="redo-sess", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude",
                          status="done", session_id="redo-sess", start_time=6000.0)
                # Pre-existing done collect state
                state.set(task.id, "collect", model="claude", status="done")
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state, force=True)

        self.assertTrue(result)
        info = state.get(task.id, "collect", "claude")
        self.assertTrue(info["forced"])


class CollectRecoveryErrorPathsTest(unittest.TestCase):
    """Recovery flag must be set in state even when collect fails during salvage/force."""

    def test_salvage_no_jsonl_marks_skipped_with_recovery(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / "run_claude"
            run_dir.mkdir()

            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude",
                          status="failed", session_id="s1", start_time=1000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=None):
                    result = collect_single(task, "claude", config, state, salvage=True)

        self.assertFalse(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "skipped")
        self.assertTrue(info["recovery"])
        self.assertIn("salvage", info["error"])

    def test_salvage_run_dir_missing_marks_failed_with_recovery(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir="/nonexistent/run/CT-0001")
                state.set(task.id, "run", model="claude",
                          status="failed", session_id="s1", start_time=1000.0)
                state.save()

                result = collect_single(task, "claude", config, state, salvage=True)

        self.assertFalse(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "failed")
        self.assertTrue(info["recovery"])

    def test_force_run_dir_missing_marks_failed_with_recovery(self) -> None:
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir="/nonexistent/run/CT-0001")
                state.set(task.id, "run", model="claude", status="failed")
                state.save()

                result = collect_single(task, "claude", config, state, force=True)

        self.assertFalse(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "failed")
        self.assertTrue(info["recovery"])

    def test_normal_no_jsonl_has_no_recovery(self) -> None:
        """Normal mode + no JSONL → failed WITHOUT recovery."""
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            run_dir = tmp / "run_claude"
            run_dir.mkdir()

            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude",
                          status="done", session_id="s1", start_time=1000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=None):
                    result = collect_single(task, "claude", config, state)

        self.assertFalse(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "failed")
        self.assertFalse(info.get("recovery", False))

    def test_normal_session_mismatch_has_no_recovery(self) -> None:
        """Normal mode + session_id mismatch → failed WITHOUT recovery."""
        task = make_task(bad_pattern="lazy_shortcut")
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            with patch.object(BatchConfig, "base_dir", new_callable=PropertyMock, return_value=tmp):
                config = build_config([task], person_id="")
                delivery_dir = config.delivery_dir
                delivery_dir.mkdir(parents=True, exist_ok=True)

                run_dir = tmp / "run_claude"
                run_dir.mkdir()

                traj_jsonl = tmp / "wrong.jsonl"
                _write_jsonl(traj_jsonl, session_id="wrong-sid", model="claude-3.5-sonnet-20241022")

                state = PipelineState(delivery_dir / "pipeline_state.json")
                state.set(task.id, "prepare", claude_dir=str(run_dir))
                state.set(task.id, "run", model="claude",
                          status="done", session_id="expected-sid", start_time=1000.0)
                state.save()

                with patch("ctpipe.collect.find_trajectory_for_run", return_value=traj_jsonl):
                    result = collect_single(task, "claude", config, state)

        self.assertFalse(result)
        info = state.get(task.id, "collect", "claude")
        self.assertEqual(info["status"], "failed")
        self.assertIn("session_id mismatch", info["error"])
        self.assertFalse(info.get("recovery", False))


if __name__ == "__main__":
    unittest.main()
